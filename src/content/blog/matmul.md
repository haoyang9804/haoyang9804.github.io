---
title: "matmul.cu"
description: "从 naive matmul 开始，一步步观察 CUDA 矩阵乘 kernel 的性能瓶颈和优化方向。"
pubDate: 2026-06-02
tags: ["cuda", "llm-infra", "kernel"]
draft: false
---

# matmul.cu

matmul可谓入门算子的基础中的基础。但即便基础，matmul也有不少优化。本文旨在通过探讨这个基础算子的几种写法，从最朴素的写法一步一步进行优化。

## 前言

### 阅读本文需要什么基础？
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

### 一些写算子的思路
```
Kernel
 ├── Block 0
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 ├── Block 1
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 └── ...
```
上图展示的是CUDA的执行逻辑，我们写kernel时其实是在思考一个block会做哪些事，即每个block的thread在做什么，以及这个block中其他thread正在做什么。因此，初入算子时需要锻炼一种全局视角，要考虑一个warp/block下的其他thread做了啥，他们的结果该如何汇总，等等。接下来进阶版本的matmul会用到这种思想。

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

## 朴素的不行，得玩点花活
上面的kernel最大的问题是arithmetic intensity (AI) 不够
$$
AI = \frac{计算量}{\text{GPU 的 HBM（High Bandwidth Memory）读写的数据总量}}\ \ \text{FLOPs} / \text{Byte}
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
这是一个很糟糕的数据。下表展示了近几代NV GPU的的$AI$。其中B200的peak tflops数据来自[8x NVIDIA Blackwell SXM](https://www.nvidia.com/en-us/data-center/hgx/?utm_source=chatgpt.com)的估算：8卡TF32 Tensor Core为18 PFLOPS with sparsity，因此，单卡约为 $18000/8=2250\ \text{TFLOPS}$

| GPU      |                                  Peak TFLOPS | HBM Bandwidth | $AI$ critical |
| -------- | -------------------------------------------: | ------------: | ------------: |
| A100 80G |   312 TFLOPS(TF32 Tensor Core with sparsity) |      2.0 TB/s |          ≈156 |
| H100 SXM |  989 TFLOPS (TF32 Tensor Core with sparsity) |     3.35 TB/s |          ≈295 |
| B200     | 2250 TFLOPS (TF32 Tensor Core with sparsity) |      8.0 TB/s |          ≈281 |

这些卡的$AI$都远高$\frac{1}{2}$。
一个
