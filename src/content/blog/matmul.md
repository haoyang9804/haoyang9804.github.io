---
title: "matmul.cu"
description: "从 naive matmul 开始，一步步观察 CUDA 矩阵乘 kernel 的性能瓶颈和优化方向。"
pubDate: 2026-06-02
tags: ["cuda", "llm-infra", "kernel"]
draft: false
---

# matmul.cu

matmul可谓入门算子的基础中的基础。但即便基础，matmul也有不少优化。本文旨在通过探讨这个基础算子的几种写法，从最朴素的写法一步一步进行优化。

## 阅读本文需要什么基础？
本文是写给算子新手的，也是我作为新手写给自己用于复习总结的。
只要你懂如下代码在做啥，你就可以理解本文：
```cpp
#include <cuda_runtime.h>
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N){
        C[tid] = A[tid] + B[tid];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

```

## 题目描述

题目来自leetgpu。
矩阵$A$的shape为$M\times N$ ，矩阵B的shape为 $N\times K$, 他们都是float32 dtype，计算他们的矩阵乘并存入C。

## 先来个朴素的
```c++
#include <cuda_runtime.h>

constexpr int kTileM = 16;
constexpr int kTileN = 16; // reduce
constexpr int kTileK = 16;

__host__ __device__ static inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    for (int row_block = blockIdx.x; row_block < ceil_div_int(M, kTileM); row_block += gridDim.x) {
        for (int col_block = blockIdx.y; col_block < ceil_div_int(K, kTileK); col_block += gridDim.y) {
            int row = row_block * kTileM + threadIdx.x;
            int col = col_block * kTileK + threadIdx.y;
            if (row >= M || col >= K) {
                continue;
            }
            float sum = 0.0f;
            for (int k = 0; k < N; k += 1) {
                sum += A[row * N + k] * B[k * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(kTileM, kTileK);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (K + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

```
上述kernel对应的矩阵乘方式如下图所示

![Naive matmul tile mapping](../pics/matmul-naive-tile.png)

`threadIdx.x`对应这矩阵`A`的粉色行，`threadIdx.y`对应这矩阵`B的绿色列，在leetgpu上耗时

![Naive matmul LeetGPU timing](../pics/matmul-naive-timing.png)

性能极差 :(
其中最容易发现的影响性能的操作是`threadidx.x`遍历`A`的行。
全局`thread id = threadIdx.x + blockDim.x * threadIdx.y`，因此，同一个 warp 内相邻线程访问 A 的地址跨度为 N 个元素（row stride），无法形成 coalesced memory access，因此产生大量全局内存事务，性能极差。
我们只需要调换`threadIdx.x`和`threadIdx.y`，就可以解决这个问题：

![Coalesced matmul tile mapping](../pics/matmul-coalesced-tile.png)

```cpp
#include <cuda_runtime.h>

constexpr int kTileM = 16;
constexpr int kTileN = 16; // reduce
constexpr int kTileK = 16;

__host__ __device__ static inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    for (int row_block = blockIdx.y; row_block < ceil_div_int(M, kTileM); row_block += gridDim.y) {
        for (int col_block = blockIdx.x; col_block < ceil_div_int(K, kTileK); col_block += gridDim.x) {
            int row = row_block * kTileM + threadIdx.y;
            int col = col_block * kTileK + threadIdx.x;
            if (row >= M || col >= K) {
                continue;
            }
            float sum = 0.0f;
            for (int k = 0; k < N; k += 1) {
                sum += A[row * N + k] * B[k * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(kTileM, kTileK);
    dim3 blocksPerGrid((M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (K + threadsPerBlock.x - 1) / threadsPerBlock.x);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

```

![Coalesced matmul LeetGPU timing](../pics/matmul-coalesced-timing.png)

效果拔群 :)
但从这个percentile来看，还有很大进步空间。

## 朴素的不行，得玩点花活 - shared memory tiling
上面的kernel最大的问题是arithmetic intensity (AI) 不够。

$$
\mathrm{AI}
=
\frac{
  \text{计算量}
}{
  \text{GPU HBM 读写的数据总量}
}
\quad
(\text{FLOPs}/\text{Byte})
$$

kernel中的compute代码
```go
for (int k = 0; k < N; k += 1) {
	sum += A[row * N + k] * B[k * K + col];
}
```
每一次`sum += A[row * N + k] * B[k * K + col];`的flops为2(一个Fused Multiply-Add(FMA)的flops为2)；且由于A和B都是float32，每个元素4 bytes。
根据上述公式可得，每个thread的AI为
$$
AI = \frac{2N}{4N} = \frac{1}{2}
$$
这是一个很糟糕的数据。下表展示了近几代NV GPU的$AI$ critical。Tensor Core列使用TF32 Tensor Core with sparsity的peak，CUDA Core列使用FP32 peak。

| GPU      | TF32 Tensor Core Peak | FP32 CUDA Core Peak | HBM Bandwidth | Tensor Core $AI$ critical | CUDA Core $AI$ critical |
| -------- | --------------------: | ------------------: | ------------: | ------------------------: | ----------------------: |
| A100 80G SXM | 312 TFLOPS | 19.5 TFLOPS | 2.039 TB/s | ≈153 | ≈9.6 |
| H100 SXM | 989 TFLOPS | 67 TFLOPS | 3.35 TB/s | ≈295 | ≈20 |
| B200 SXM | 2250 TFLOPS | 75 TFLOPS | 8.0 TB/s | ≈281 | ≈9.4 |

当前这个kernel使用的是CUDA Core上的FP32 FMA，并没有使用Tensor Core；因此分析它是否memory bound时，应该主要看上表中的CUDA Core $AI$ critical。

即便只看CUDA Core $AI$ critical，这些卡也都远高$\frac{1}{2}$，因此上述两个kernel是memory bound的。
在阐述如何提高$AI$以及为什么要提高$AI$前，我们先看看$AI$代表着啥。

下图展示了单卡上各个存储模块的带宽和延迟。其中，顺序大量访问取决于带宽水平，随机零散访问取决于延迟。图中可见，HBM这两个维度都远弱于上层存储模块，且HBM是启动kernel后，矩阵被放置的位置。
因此，从HBM读写往往是kernel性能的瓶颈。
<span id="fig-gpu-bandwidth-hierarchy"></span>

![GPU bandwidth hierarchy](../pics/GPU-bandwidth-hierarchy.png)

上述两个kernel当前的$AI$低的一大原因是从HBM load一个数据做一次计算。然而矩阵$A$中每一个数据都会和多个数据做计算，比如$A[i][j]$会和$B[j][:]$做乘法。
因此，一个符合直觉的做法是把$A[i][j]$从HBM load到shared memory，在block内部的每个`thread id`对应的$B[j][k]$会从shared memory而非HBM load $A[i][j]$.
这种做法就是所谓的shared memory tiling。在写此类优化时，我们应该把load和compute分开：先考虑如何在一个block中以每个thread为最基本的“劳动个体”，让他们合力把矩阵块从HBM移动到shared memory；再考虑每个thread如何基于shared memory存的矩阵块进行计算。

如下图所示，在load的时候，每个thread都沿着reduce的方向(也就是$N$的方向)把几个数据load到shared memory。
图中绿色/粉色不同的深浅代表着不同的load轮数。即，每一次load都会沿着$B$的列load两个数据，沿着$A$的行load两个数据，
并且，一个block内的全部thread同时发力进行load，图中画出了相邻的两个thread load同一行$A$，相邻列$B$的场景。
![shared memory tiling - load](../pics/matmul-shared-memory-load.png)
假设一个block中`blockDim.x=2`, `$blockDim.y=2`，那么load完后的状态应该是
![](../pics/matmul-shared-memory-after-load.png)
随后compute就在每次load后做局部计算即可，计算流程如图所示，每个`thread id`处理一个shared memory中矩阵块的reduce

<span id="fig-matmul-shared-memory-compute"></span>
![](../pics/matmul-shared-memory-compute.png)

<span id="shared-memory-kernel-code"></span>

代码如下：
```cpp
#include <cuda_runtime.h>

constexpr int kTileM = 32;
constexpr int kTileN = 32; // reduce
constexpr int kTileK = 32;

__host__ __device__ static inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

template <int m_block_size, int n_tile_size, int k_block_size>
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // shared memory
    __shared__ float a_tile[m_block_size][n_tile_size];
    __shared__ float b_tile[n_tile_size][k_block_size];
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    for (int row_block = blockIdx.x; row_block < ceil_div_int(M, m_block_size); row_block += gridDim.x) {
        for (int col_block = blockIdx.y; col_block < ceil_div_int(K, k_block_size); col_block += gridDim.y) {
            int row = row_block * m_block_size + local_row;
            int col = col_block * k_block_size + local_col;
            float sum = 0.0f;

            int total_threads = blockDim.x * blockDim.y;
            int tid = threadIdx.x + blockDim.x * threadIdx.y;
            for (int k0 = 0; k0 < N; k0 += n_tile_size) {
                // load
                for (int idx = tid; idx < m_block_size * n_tile_size; idx += total_threads) {
                    int a_row = idx / n_tile_size;
                    int a_col = idx % n_tile_size;
                    int global_a_row = row_block * m_block_size + a_row;
                    int global_a_col = k0 + a_col;
                    a_tile[a_row][a_col] =
                        (global_a_row < M && global_a_col < N) ? A[global_a_row * N + global_a_col] : 0.0f;
                }
                for (int idx = tid; idx < n_tile_size * k_block_size; idx += total_threads) {
                    int b_row = idx / k_block_size;
                    int b_col = idx % k_block_size;
                    int global_b_row = k0 + b_row;
                    int global_b_col = col_block * k_block_size + b_col;
                    b_tile[b_row][b_col] =
                        (global_b_row < N && global_b_col < K) ? B[global_b_row * K + global_b_col] : 0.0f;
                }
                __syncthreads();
                // compute
                #pragma unroll
                for (int k = 0; k < n_tile_size; ++k) {
                    sum += a_tile[local_row][k] * b_tile[k][local_col];
                }
                __syncthreads();
            }

            if (row < M && col < K) {
                C[row * K + col] = sum;
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    dim3 threadsPerBlock(kTileK, kTileM);
    dim3 blocksPerGrid(ceil_div_int(M, kTileM),
                       ceil_div_int(K, kTileK));

    matrix_multiplication_kernel<kTileM, kTileN, kTileK><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

```

这个版本的kernel的$AI$有大幅提升

$$
\begin{aligned}
AI
&=
\frac{
 m\_block\_size \times n\_tile\_size \times k\_block\_size \times 2
}{
  (m\_block\_size \times n\_tile\_size
  + k\_block\_size \times n\_tile\_size)
  \times 4
} \\
&=
\frac{32 \times 32 \times 32 \times 2}{ (32 \times 32 + 32 \times 32) \times 4} \\
&= 8
\end{aligned}
$$

远优秀于之前的$\frac{1}{2}$。

这个优化在性能上有大幅提升。
![shared-memory matmul LeetGPU timing](../pics/matmul-shared-memory-timing.png)

一个题外话，上面的代码还有一些优化空间。
我们看以下全局设置。
```cpp
constexpr int kTileM = 32;
constexpr int kTileN = 32; // reduce
constexpr int kTileK = 32;
```
由于$kTileM*kTileK=1024$，已经到达了每个kernel的block size的上限。但`kTileN`可以继续上调，上调不会影响thread数量，也不会让每个thread做的计算变多，但单次从HBM load到shared memory的矩阵块变大。
| kTileN | time(ms)
|------|-----|
| 32 | 113.7 |
| 64 | 111.5 |
| 128 | 110.28 |
| 256 | compilation failed: uses too much shared data |

由上表可见，增大`kTileN`只能带来小幅性能提升：在`kTileM`和`kTileK`固定时，理想HBM AI并不会因为reduce tile变厚而线性提高，因为计算量和A/B tile load量都会随`kTileN`同时增长。
它真正改善的是外层`k0`循环次数、`__syncthreads()`次数和load loop调度开销的摊销，所以32到128会变快但幅度有限。
继续增大到256时，`a_tile[32][256] + b_tile[256][32]`需要约64KB shared memory，超过或逼近单block可用shared memory，因此编译失败。


## 还能优化 - register tiling

所谓shared memory tiling，就是把原本反复从HBM读取的数据，先搬到Shared Memory，再让很多线程重复使用。
但根据[该图](#fig-gpu-bandwidth-hierarchy), register的读写要比shared memory快得多。

[shared memory tiling](#shared-memory-kernel-code)的代码在compute时，每一次从shared memory读取数据到register后，都只计算一个sum，也就是$A$的某一行的一部分和$B$的某一列的一部分的reduce: `sum += a_tile[local_row][k] * b_tile[k][local_col];` (见[此图](#fig-matmul-shared-memory-compute))。之后，这部分register的数据就会被下一次运行覆盖，也就是说，register的数据复用很差。

因此，一个符合直觉的做法是，对于load进register的数据，我们多用几次，多算几轮。为此，我们可以把`sum += a_tile[local_row][k] * b_tile[k][local_col];`升级为
```cpp
for (int thread_row = 0; thread_row < thread_tile_m; ++thread_row) {
    float a_value = a_tile[local_row_start + thread_row][k];
    #pragma unroll
    for (int thread_col = 0; thread_col < thread_tile_k; ++thread_col) {
        accum[thread_row][thread_col] +=
            a_value * b_tile[k][local_col_start + thread_col];
    }
}
```          
这里，`thread_tile_m`表示的是一个thread要在$M$的维度上处理多少个数据。如下图所示，每个thread负责把矩阵$A$中存储在shared memory的矩阵块(粉色)的两行(粗黑框网格填充)和矩阵$B$中存储在shared memory的矩阵块(绿色)的两列(粗黑框网格填充)先load进入register，然后两两reduce。

![](../pics/matmul-register-tiling-compute.png)

同时需要注意的是，register tiling并没有提升$AI$，它改变的是片上存储层级里的数据复用方式，提高了数据复用率。

偏题一下，单个block内通过shared memory tiling能达到的HBM $AI$会受片上资源限制；想继续提高$AI$，需要增大输出tile，让一次load进来的矩阵块参与更多FMA，但这会很快受到shared memory、register和thread数量的约束。在这种情况下，可以通过 1) 换更小的数据类型； 2) 融合算子，避免中间结果回到HBM；3) 利用L2 cache的跨block复用等方式提升$AI$.

代码如下：
```cpp
#include <cuda_runtime.h>
#include <algorithm>

constexpr int kTileM = 32;
constexpr int kTileN = 32; // reduce
constexpr int kTileK = 32;
constexpr int kThreadTileM = 4;
constexpr int kThreadTileK = 4;

__host__ __device__ static inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

template <int m_block_size, int n_tile_size, int k_block_size, int thread_tile_m, int thread_tile_k>
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    static_assert(m_block_size % thread_tile_m == 0, "thread_tile_m must divide m_block_size");
    static_assert(k_block_size % thread_tile_k == 0, "thread_tile_k must divide k_block_size");
    __shared__ float a_tile[m_block_size][n_tile_size];
    __shared__ float b_tile[n_tile_size][k_block_size];
    int threads = blockDim.x * blockDim.y;
    for (int c_row_block = blockIdx.y; c_row_block < ceil_div_int(M, m_block_size); c_row_block += gridDim.y) {
        for (int c_col_block = blockIdx.x; c_col_block < ceil_div_int(K, k_block_size); c_col_block += gridDim.x) {
           int local_row_start = threadIdx.y * thread_tile_m;
           int local_col_start = threadIdx.x * thread_tile_k;
           int tid = threadIdx.x + blockDim.x * threadIdx.y;
           int local_A_block_size = m_block_size * n_tile_size;
           int local_B_block_size = k_block_size * n_tile_size;
           float accum[thread_tile_m][thread_tile_k] = {0.0f};
           #pragma unroll
           for (int n = 0; n < N; n += n_tile_size) {
                // load into shared memory
                for (int i = tid; i < local_A_block_size; i += threads) {
                    int a_tile_col = i % n_tile_size;
                    int a_tile_row = i / n_tile_size;
                    int global_a_row = c_row_block * m_block_size + a_tile_row;
                    int global_a_col = n + a_tile_col;
                    a_tile[a_tile_row][a_tile_col] =
                        (global_a_row < M && global_a_col < N) ? A[global_a_row * N + global_a_col] : 0.0f;
                }
                for (int i = tid; i < local_B_block_size; i += threads) {
                    int b_tile_col = i % k_block_size;
                    int b_tile_row = i / k_block_size;
                    int global_b_row = n + b_tile_row;
                    int global_b_col = c_col_block * k_block_size + b_tile_col;
                    b_tile[b_tile_row][b_tile_col] =
                        (global_b_row < N && global_b_col < K) ? B[global_b_row * K + global_b_col] : 0.0f;
                }
                __syncthreads();
                // load into register and compute
                #pragma unroll
                for (int k = 0; k < n_tile_size; ++k) {
                    #pragma unroll
                    for (int thread_row = 0; thread_row < thread_tile_m; ++thread_row) {
                        float a_value = a_tile[local_row_start + thread_row][k];
                        #pragma unroll
                        for (int thread_col = 0; thread_col < thread_tile_k; ++thread_col) {
                            accum[thread_row][thread_col] +=
                                a_value * b_tile[k][local_col_start + thread_col];
                        }
                    }
                }
                __syncthreads();
           }
           #pragma unroll
           for (int thread_row = 0; thread_row < thread_tile_m; ++thread_row) {
               int c_row = c_row_block * m_block_size + local_row_start + thread_row;
               #pragma unroll
               for (int thread_col = 0; thread_col < thread_tile_k; ++thread_col) {
                   int c_col = c_col_block * k_block_size + local_col_start + thread_col;
                   if (c_row < M && c_col < K) {
                       C[c_row * K + c_col] = accum[thread_row][thread_col];
                   }
               }
           }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(kTileK / kThreadTileK, kTileM / kThreadTileM);
    dim3 blocksPerGrid(ceil_div_int(K, kTileK), ceil_div_int(M, kTileM));

    matrix_multiplication_kernel<kTileM, kTileN, kTileK, kThreadTileM, kThreadTileK>
        <<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

```

看一眼性能，进步了一大截 :)
![](../pics/matmul-register-tiling-timing.png)
但还有进步空间。目前每个block的block size为

$$
\frac{\text{kTileM}}{\text{kThreadTileM}} \times \frac{\text{kTileK}}{\text{kThreadTileK}}=64
$$

如果把
```cpp
constexpr int kTileM = 32;
constexpr int kTileN = 32; // reduce
constexpr int kTileK = 32;
constexpr int kThreadTileM = 4;
constexpr int kThreadTileK = 4;
```
改为
```cpp
constexpr int kTileM = 128;
constexpr int kTileN = 4; // reduce
constexpr int kTileK = 128;
constexpr int kThreadTileM = 8;
constexpr int kThreadTileK = 8;
```

那么，$AI$提升，从$8$提升到$32$。 block size从64提升到256,也可以提升occupancy，SMA和SMI。
做了这个简单的参数变化后，性能又提升一大截：
![](../pics/matmul-register-tiling-timing2.png)

小插曲：如果
```cpp
constexpr int kTileM = 128;
constexpr int kTileN = 4; // reduce
constexpr int kTileK = 128;
constexpr int kThreadTileM = 4;
constexpr int kThreadTileK = 4;
```
那么，AI还是$32$，但是block size为1024(已到单kernel上限)，且每个thread的计算量大大缩小。
每个thread的计算量为 $\text{kThreadTileM}\times\text{kThreadTileK}\times\text{kTileN}$，因此计算量从256下降到了64。
thread变多，但每个thread计算量变少。每个 thread 的计算变薄后，`__syncthreads()` 和 warp 调度这些固定开销更难被计算量摊销。

这些问题导致了性能的大幅度劣化：
![](../pics/matmul-register-tiling-timing3.png)

还有一个很有趣的现象：

在固定`kTileM=128`、`kTileK=128`、`kThreadTileM=8`、`kThreadTileK=8`时，继续调`kTileN`不会改变理想$AI$，因为每个block负责的输出tile大小没有变，`kTileN`只是在改变每轮reduce staging进shared memory的厚度。

| kTileN | shared memory / block | 每轮每个thread的FMA数量 | 外层reduce轮数 | time(ms) |
| ------ | --------------------- | ----------------------- | -------------- | -------- |
| 4 | 4096 B | $8\times8\times4=256$ | $\lceil N/4\rceil$ | 49.93 |
| 8 | 8192 B | $8\times8\times8=512$ | $\lceil N/8\rceil$ | 51.40 |
| 16 | 16384 B | $8\times8\times16=1024$ | $\lceil N/16\rceil$ | 51.40 |
| 32 | 32768 B | $8\times8\times32=2048$ | $\lceil N/32\rceil$ | 50.66 |

按直觉，`kTileN`变大后，外层循环和`__syncthreads()`次数减少，似乎应该更快；但在这个kernel里，`kTileN=4`的每轮计算量已经足够摊销同步开销，而更大的`kTileN`并不会提升$AI$，只会增加shared memory占用和单轮展开后的调度压力，所以最终`kTileN=4`反而是更好的平衡点。
