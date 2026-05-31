---
title: "现代孔乙己 - softmax 的 N 种 CUDA 写法"
description: "一次从 naive softmax 到多 kernel reduce 的 CUDA kernel 性能实验笔记。"
pubDate: 2026-05-22
tags: ["cuda", "llm-infra", "kernel"]
draft: false
---

本次实验旨在对kernel性能有一个初步的了解，锻炼看到性能大概猜到瓶颈在哪里的能力。

实验都在单卡H20上完成。H20有78个SM，max_warps_per_sm = 64，max_blocks_per_sm = 32，registers_per_sm = 65536, 这些数据与下述粗粒度的简单实验高度相关。

# 背景

初始的softmax$\frac{e^{s_i-max_{k=0}^N{s_k}}}{\sum_{j=0}^Ne^{s_j-max_{k=0}^N{s_k}}}$无法通过kernel高效解决，这是因为单次 online softmax kernel 内，reduce 的协作范围通常最多到 block 级别。因此首先想到的就是分三个kernel，第一个kernel（称之为pass1）计算每个block的 $max_{k=B_i}^{B_{i+1}-1}{s_k}$和 $\sum_{j=B_i}^{B_{i+1}-1}e^{s_j-max_{k=B_i}^{B_{i+1}-1}{s_k}}$，第二个kernel（称之为pass2）把block-level summary整合成global summary $e^{s_i-max_{k=0}^N{s_k}}$和 $\sum_{j=0}^Ne^{s_j-max_{k=0}^N{s_k}}$，然后第三个kernel （pass3）做最简单的事：把input[i] 通过计算映射到output[i]。

# 代码

cuda代码

```cpp
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <float.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

enum TestCaseId {
    kCaseSine = 0,
    kCaseZeros = 1,
    kCaseRamp = 2,
    kCaseReverse = 3,
    kCaseAlternating = 4,
    kCaseSpike = 5,
    kCaseRandom = 6,
};

struct SolveConfig {
    int pass1_stride;
    int pass2_stride;
    int pass3_stride;
    int pass1_tile_cap;
    int pass2_tile_cap;
    int pass3_tile_cap;
};

__host__ __device__ __forceinline__ float input_value_at(int i, int N, int test_case) {
    switch (test_case) {
        case kCaseSine:
            return sinf(i * 0.001f) * 4.0f + cosf(i * 0.0001f);
        case kCaseZeros:
            return 0.0f;
        case kCaseRamp: {
            int denom = N > 1 ? N - 1 : 1;
            return -8.0f + 16.0f * static_cast<float>(i) / static_cast<float>(denom);
        }
        case kCaseReverse: {
            int denom = N > 1 ? N - 1 : 1;
            return 8.0f - 16.0f * static_cast<float>(i) / static_cast<float>(denom);
        }
        case kCaseAlternating:
            return (i & 1) ? -8.0f : 8.0f;
        case kCaseSpike:
            if (i == N / 2) {
                return 12.0f;
            }
            return (i % 10007 == 0) ? 8.0f : -8.0f;
        case kCaseRandom: {
            unsigned int state = 123456789u ^ static_cast<unsigned int>(i * 747796405u);
            state ^= state >> 16;
            state *= 2246822519u;
            state ^= state >> 13;
            state *= 3266489917u;
            state ^= state >> 16;
            float unit = static_cast<float>(state & 0x00ffffffu) / 16777216.0f;
            return -8.0f + 16.0f * unit;
        }
        default:
            return 0.0f;
    }
}

__device__ __forceinline__ void combine_softmax_pair(
    float& esum, float& mx, float other_esum, float other_mx) {
    if (other_esum == 0.0f) {
        return;
    }
    if (esum == 0.0f) {
        esum = other_esum;
        mx = other_mx;
        return;
    }
    float new_mx = fmaxf(mx, other_mx);
    esum = esum * __expf(mx - new_mx) + other_esum * __expf(other_mx - new_mx);
    mx = new_mx;
}

__device__ __forceinline__ void add_softmax_value(float& esum, float& mx, float val) {
    if (esum == 0.0f) {
        esum = 1.0f;
        mx = val;
        return;
    }
    float new_mx = fmaxf(mx, val);
    esum = esum * __expf(mx - new_mx) + __expf(val - new_mx);
    mx = new_mx;
}

__device__ __forceinline__ void warp_reduce(float& esum, float& mx) {
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        float other_esum = __shfl_down_sync(mask, esum, offset);
        float other_mx = __shfl_down_sync(mask, mx, offset);
        combine_softmax_pair(esum, mx, other_esum, other_mx);
    }
}

template <int ThreadsPerBlock>
__device__ __forceinline__ void block_reduce(float& esum, float& mx) {
    static_assert(ThreadsPerBlock % kWarpSize == 0, "ThreadsPerBlock must be a multiple of warp size");
    constexpr int warp_num = ThreadsPerBlock / kWarpSize;
    __shared__ float sdata[warp_num * 2];

    warp_reduce(esum, mx);
    int warp_id = threadIdx.x / kWarpSize;
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        sdata[warp_id] = esum;
        sdata[warp_id + warp_num] = mx;
    }
    __syncthreads();

    if (threadIdx.x < kWarpSize) {
        esum = threadIdx.x < warp_num ? sdata[threadIdx.x] : 0.0f;
        mx = threadIdx.x < warp_num ? sdata[threadIdx.x + warp_num] : -FLT_MAX;
        warp_reduce(esum, mx);
    }
}

template <int ThreadsPerBlock>
__global__ void init_input_kernel(float* input, int N, int test_case) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += grid_stride) {
        input[i] = input_value_at(i, N, test_case);
    }
}

template <int BlockStride, int ThreadsPerBlock>
__global__ void softmax_pass1(
    const float* input, float* block_esum, float* block_mx, int N) {
    constexpr int tile_width = ThreadsPerBlock * BlockStride;
    int base0 = blockIdx.x * tile_width + threadIdx.x;
    int grid_step = gridDim.x * tile_width;
    float mx = -FLT_MAX;
    float esum = 0.0f;

    for (int base = base0; base < N; base += grid_step) {
        #pragma unroll 1
        for (int j = 0; j < BlockStride; ++j) {
            int cur = base + j * ThreadsPerBlock;
            if (cur < N) {
                add_softmax_value(esum, mx, input[cur]);
            }
        }
    }

    block_reduce<ThreadsPerBlock>(esum, mx);
    if (threadIdx.x == 0) {
        block_esum[blockIdx.x] = esum;
        block_mx[blockIdx.x] = mx;
    }
}

template <int BlockStride, int ThreadsPerBlock>
__global__ void softmax_pass2(
    const float* input_esum, const float* input_mx,
    float* output_esum, float* output_mx, int count) {
    constexpr int tile_width = ThreadsPerBlock * BlockStride;
    int base0 = blockIdx.x * tile_width + threadIdx.x;
    int grid_step = gridDim.x * tile_width;
    float mx = -FLT_MAX;
    float esum = 0.0f;

    for (int base = base0; base < count; base += grid_step) {
        #pragma unroll 1
        for (int j = 0; j < BlockStride; ++j) {
            int cur = base + j * ThreadsPerBlock;
            if (cur < count) {
                combine_softmax_pair(esum, mx, input_esum[cur], input_mx[cur]);
            }
        }
    }

    block_reduce<ThreadsPerBlock>(esum, mx);
    if (threadIdx.x == 0) {
        output_esum[blockIdx.x] = esum;
        output_mx[blockIdx.x] = mx;
    }
}

template <int BlockStride, int ThreadsPerBlock>
__global__ void softmax_pass3(
    const float* input, float* output, int N, const float* final_esum, const float* final_mx) {
    constexpr int tile_width = ThreadsPerBlock * BlockStride;
    int base0 = blockIdx.x * tile_width + threadIdx.x;
    int grid_step = gridDim.x * tile_width;
    float mx = final_mx[0];
    float esum = final_esum[0];

    for (int base = base0; base < N; base += grid_step) {
        #pragma unroll
        for (int j = 0; j < BlockStride; ++j) {
            int cur = base + j * ThreadsPerBlock;
            if (cur < N) {
                output[cur] = __expf(input[cur] - mx) / esum;
            }
        }
    }
}

static int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static int tile_count_for(int count, int block_stride) {
    return ceil_div_int(count, kThreadsPerBlock * block_stride);
}

static int grid_for_count(int count, int block_stride, int tile_cap) {
    int tiles = tile_count_for(count, block_stride);
    if (tile_cap > 0 && tiles > tile_cap) {
        return tile_cap;
    }
    return tiles > 0 ? tiles : 1;
}

static bool supported_stride(int stride) {
    switch (stride) {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
        case 32:
        case 64:
            return true;
        default:
            return false;
    }
}

#define DISPATCH_STRIDE(stride, ...) \
    do {                             \
        switch (stride) {            \
            case 1: { constexpr int kStride = 1; __VA_ARGS__; break; }   \
            case 2: { constexpr int kStride = 2; __VA_ARGS__; break; }   \
            case 4: { constexpr int kStride = 4; __VA_ARGS__; break; }   \
            case 8: { constexpr int kStride = 8; __VA_ARGS__; break; }   \
            case 16: { constexpr int kStride = 16; __VA_ARGS__; break; } \
            case 32: { constexpr int kStride = 32; __VA_ARGS__; break; } \
            case 64: { constexpr int kStride = 64; __VA_ARGS__; break; } \
            default:                                                      \
                std::fprintf(stderr, "Unsupported block stride: %d\n", stride); \
                std::exit(EXIT_FAILURE);                                  \
        }                                                                 \
    } while (0)

static void launch_pass1(
    const float* input, float* block_esum, float* block_mx,
    int N, int block_stride, int grid_x) {
    DISPATCH_STRIDE(block_stride, {
        softmax_pass1<kStride, kThreadsPerBlock>
            <<<grid_x, kThreadsPerBlock>>>(input, block_esum, block_mx, N);
    });
}

static void launch_pass2(
    const float* in_esum, const float* in_mx,
    float* out_esum, float* out_mx,
    int count, int block_stride, int grid_x) {
    DISPATCH_STRIDE(block_stride, {
        softmax_pass2<kStride, kThreadsPerBlock>
            <<<grid_x, kThreadsPerBlock>>>(in_esum, in_mx, out_esum, out_mx, count);
    });
}

static void launch_pass3(
    const float* input, float* output, int N,
    const float* final_esum, const float* final_mx,
    int block_stride, int grid_x) {
    DISPATCH_STRIDE(block_stride, {
        softmax_pass3<kStride, kThreadsPerBlock>
            <<<grid_x, kThreadsPerBlock>>>(input, output, N, final_esum, final_mx);
    });
}

static void solve_grid_block(
    const float* input,
    float* output,
    int N,
    const SolveConfig& cfg,
    float* scratch_esum_a,
    float* scratch_mx_a,
    float* scratch_esum_b,
    float* scratch_mx_b) {
    int pass1_grid = grid_for_count(N, cfg.pass1_stride, cfg.pass1_tile_cap);
    launch_pass1(input, scratch_esum_a, scratch_mx_a, N, cfg.pass1_stride, pass1_grid);

    int count = pass1_grid;
    float* in_esum = scratch_esum_a;
    float* in_mx = scratch_mx_a;
    float* out_esum = scratch_esum_b;
    float* out_mx = scratch_mx_b;

    while (count > 1) {
        int reduce_grid = grid_for_count(count, cfg.pass2_stride, cfg.pass2_tile_cap);
        launch_pass2(
            in_esum, in_mx, out_esum, out_mx, count, cfg.pass2_stride, reduce_grid);
        count = reduce_grid;
        std::swap(in_esum, out_esum);
        std::swap(in_mx, out_mx);
    }

    int pass3_grid = grid_for_count(N, cfg.pass3_stride, cfg.pass3_tile_cap);
    launch_pass3(input, output, N, in_esum, in_mx, cfg.pass3_stride, pass3_grid);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void cpu_softmax(const std::vector<float>& input, std::vector<float>& output) {
    float mx = *std::max_element(input.begin(), input.end());
    double esum = 0.0;
    for (float x : input) {
        esum += std::exp(static_cast<double>(x - mx));
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(std::exp(static_cast<double>(input[i] - mx)) / esum);
    }
}

static void fill_input(std::vector<float>& input, int test_case) {
    int N = static_cast<int>(input.size());
    for (int i = 0; i < N; ++i) {
        input[i] = input_value_at(i, N, test_case);
    }
}

static void print_cases() {
    std::printf("cases: sine, zeros, ramp, reverse, alternating, spike, random\n");
}

static int parse_test_case(const char* test_case) {
    if (std::strcmp(test_case, "sine") == 0) return kCaseSine;
    if (std::strcmp(test_case, "zeros") == 0) return kCaseZeros;
    if (std::strcmp(test_case, "ramp") == 0) return kCaseRamp;
    if (std::strcmp(test_case, "reverse") == 0) return kCaseReverse;
    if (std::strcmp(test_case, "alternating") == 0) return kCaseAlternating;
    if (std::strcmp(test_case, "spike") == 0) return kCaseSpike;
    if (std::strcmp(test_case, "random") == 0) return kCaseRandom;
    return -1;
}

static bool parse_int_in_range(const char* text, int min_value, int max_value, int* value) {
    char* end = nullptr;
    errno = 0;
    double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

static bool parse_positive_int(const char* text, int* value) {
    return parse_int_in_range(text, 1, INT_MAX, value);
}

static bool parse_nonnegative_int(const char* text, int* value) {
    return parse_int_in_range(text, 0, INT_MAX, value);
}

static bool parse_stride_arg(const char* text, int* value) {
    if (!parse_positive_int(text, value)) {
        return false;
    }
    return supported_stride(*value);
}

static bool parse_tile_cap_arg(const char* text, int sm_count, int* value) {
    if (std::strcmp(text, "none") == 0 || std::strcmp(text, "unlimited") == 0 ||
        std::strcmp(text, "nogrid") == 0) {
        *value = 0;
        return true;
    }
    if (std::strncmp(text, "sm", 2) == 0) {
        const char* p = text + 2;
        if (*p == '*' || *p == 'x' || *p == 'X') {
            ++p;
        }
        int multiplier = 1;
        if (*p != '\0' && !parse_positive_int(p, &multiplier)) {
            return false;
        }
        if (sm_count <= 0 || multiplier > INT_MAX / sm_count) {
            return false;
        }
        *value = sm_count * multiplier;
        return true;
    }
    return parse_nonnegative_int(text, value);
}

static void print_usage(const char* argv0) {
    std::fprintf(
        stderr,
        "Usage: %s [N] [profile_iters] [warmup_iters] [case] [auto|validate|novalidate] "
        "[pass1_stride] [pass2_stride] [pass3_stride] [pass1_tile_cap] [pass2_tile_cap] [pass3_tile_cap]\n"
        "  legacy form is also accepted: [pass1_stride] [pass3_stride] [pass1_tile_cap] [pass3_tile_cap] [pass2_tile_cap]\n"
        "  strides supported: 1,2,4,8,16,32,64\n"
        "  tile_cap: 0/none/unlimited/nogrid means no cap; sm4 or sm*4 means SM count times 4\n"
        "  defaults: pass1_stride=1 pass2_stride=32 pass3_stride=32 pass1_tile_cap=sm4 pass2_tile_cap=0 pass3_tile_cap=0\n",
        argv0);
    print_cases();
}

int main(int argc, char** argv) {
    int N = 500000;
    int profile_iters = 100;
    int warmup_iters = 10;
    const char* test_case = "sine";
    const char* validate_mode = "auto";

    if (argc > 1 && !parse_positive_int(argv[1], &N)) {
        std::fprintf(stderr, "Invalid N: %s\n", argv[1]);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc > 2 && !parse_positive_int(argv[2], &profile_iters)) {
        std::fprintf(stderr, "Invalid profile_iters: %s\n", argv[2]);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc > 3 && !parse_nonnegative_int(argv[3], &warmup_iters)) {
        std::fprintf(stderr, "Invalid warmup_iters: %s\n", argv[3]);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc > 4) {
        test_case = argv[4];
    }
    if (argc > 5) {
        validate_mode = argv[5];
    }

    int test_case_id = parse_test_case(test_case);
    if (test_case_id < 0) {
        std::fprintf(stderr, "Unknown case: %s\n", test_case);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    constexpr int kAutoValidateMaxN = 10000000;
    bool validate = N <= kAutoValidateMaxN;
    if (std::strcmp(validate_mode, "validate") == 0) {
        validate = true;
    } else if (std::strcmp(validate_mode, "novalidate") == 0) {
        validate = false;
    } else if (std::strcmp(validate_mode, "auto") != 0) {
        std::fprintf(stderr, "Invalid validate mode: %s\n", validate_mode);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int sm_count = prop.multiProcessorCount;
    SolveConfig cfg{};
    cfg.pass1_stride = 1;
    cfg.pass2_stride = 32;
    cfg.pass3_stride = 32;
    cfg.pass1_tile_cap = sm_count * 4;
    cfg.pass2_tile_cap = 0;
    cfg.pass3_tile_cap = 0;

    if (argc == 12) {
        if (!parse_stride_arg(argv[6], &cfg.pass1_stride)) {
            std::fprintf(stderr, "Invalid pass1_stride: %s\n", argv[6]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (!parse_stride_arg(argv[7], &cfg.pass2_stride)) {
            std::fprintf(stderr, "Invalid pass2_stride: %s\n", argv[7]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (!parse_stride_arg(argv[8], &cfg.pass3_stride)) {
            std::fprintf(stderr, "Invalid pass3_stride: %s\n", argv[8]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (!parse_tile_cap_arg(argv[9], sm_count, &cfg.pass1_tile_cap)) {
            std::fprintf(stderr, "Invalid pass1_tile_cap: %s\n", argv[9]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (!parse_tile_cap_arg(argv[10], sm_count, &cfg.pass2_tile_cap)) {
            std::fprintf(stderr, "Invalid pass2_tile_cap: %s\n", argv[10]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (!parse_tile_cap_arg(argv[11], sm_count, &cfg.pass3_tile_cap)) {
            std::fprintf(stderr, "Invalid pass3_tile_cap: %s\n", argv[11]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    } else if (argc <= 11) {
        if (argc > 6 && !parse_stride_arg(argv[6], &cfg.pass1_stride)) {
            std::fprintf(stderr, "Invalid pass1_stride: %s\n", argv[6]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (argc > 7 && !parse_stride_arg(argv[7], &cfg.pass3_stride)) {
            std::fprintf(stderr, "Invalid pass3_stride: %s\n", argv[7]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        cfg.pass2_stride = cfg.pass3_stride;
        if (argc > 8 && !parse_tile_cap_arg(argv[8], sm_count, &cfg.pass1_tile_cap)) {
            std::fprintf(stderr, "Invalid pass1_tile_cap: %s\n", argv[8]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (argc > 9 && !parse_tile_cap_arg(argv[9], sm_count, &cfg.pass3_tile_cap)) {
            std::fprintf(stderr, "Invalid pass3_tile_cap: %s\n", argv[9]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (argc > 10 && !parse_tile_cap_arg(argv[10], sm_count, &cfg.pass2_tile_cap)) {
            std::fprintf(stderr, "Invalid pass2_tile_cap: %s\n", argv[10]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    } else {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int pass1_tiles = tile_count_for(N, cfg.pass1_stride);
    int pass3_tiles = tile_count_for(N, cfg.pass3_stride);
    int pass1_grid = grid_for_count(N, cfg.pass1_stride, cfg.pass1_tile_cap);
    int pass3_grid = grid_for_count(N, cfg.pass3_stride, cfg.pass3_tile_cap);
    int max_summary_count = pass1_grid;

    size_t bytes = static_cast<size_t>(N) * sizeof(float);
    size_t scratch_bytes = static_cast<size_t>(max_summary_count) * sizeof(float);

    std::printf("Device: %s\n", prop.name);
    std::printf("N=%d, profile_iters=%d, warmup_iters=%d, case=%s, validate=%s\n",
                N, profile_iters, warmup_iters, test_case, validate ? "yes" : "no");
    std::printf("threads_per_block=%d, sms=%d, max_warps_per_sm=%d\n",
                kThreadsPerBlock, sm_count, prop.maxThreadsPerMultiProcessor / prop.warpSize);
    std::printf("pass1_stride=%d, pass2_stride=%d, pass3_stride=%d, pass1_tile_cap=%d, pass2_tile_cap=%d, pass3_tile_cap=%d\n",
                cfg.pass1_stride, cfg.pass2_stride, cfg.pass3_stride, cfg.pass1_tile_cap,
                cfg.pass2_tile_cap, cfg.pass3_tile_cap);
    std::printf("pass1_tiles=%d, pass1_grid=%d, pass3_tiles=%d, pass3_grid=%d\n",
                pass1_tiles, pass1_grid, pass3_tiles, pass3_grid);

    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::printf("device_memory_required_GiB=%.3f, scratch_GiB=%.6f, free_GiB=%.3f, total_GiB=%.3f\n",
                static_cast<double>(bytes * 2 + scratch_bytes * 4) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(scratch_bytes * 4) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(total_mem) / (1024.0 * 1024.0 * 1024.0));

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* scratch_esum_a = nullptr;
    float* scratch_mx_a = nullptr;
    float* scratch_esum_b = nullptr;
    float* scratch_mx_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&scratch_esum_a, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&scratch_mx_a, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&scratch_esum_b, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&scratch_mx_b, scratch_bytes));

    std::vector<float> h_input;
    if (validate) {
        h_input.resize(N);
        fill_input(h_input, test_case_id);
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        int init_blocks = tile_count_for(N, 1);
        init_blocks = init_blocks < 1024 ? init_blocks : 1024;
        init_input_kernel<kThreadsPerBlock><<<init_blocks, kThreadsPerBlock>>>(d_input, N, test_case_id);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < warmup_iters; ++i) {
        solve_grid_block(
            d_input, d_output, N, cfg, scratch_esum_a, scratch_mx_a, scratch_esum_b, scratch_mx_b);
        CUDA_CHECK(cudaGetLastError());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaProfilerStart());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < profile_iters; ++i) {
        solve_grid_block(
            d_input, d_output, N, cfg, scratch_esum_a, scratch_mx_a, scratch_esum_b, scratch_mx_b);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaProfilerStop());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / profile_iters;
    double effective_bytes = static_cast<double>(bytes) * 3.0;
    double bandwidth_gbs = effective_bytes / (avg_ms * 1.0e-3) / 1.0e9;

    std::printf("avg_time_ms=%.6f\n", avg_ms);
    std::printf("approx_effective_bandwidth_GBps=%.2f\n", bandwidth_gbs);

    float max_abs_err = 0.0f;
    if (validate) {
        std::vector<float> h_output(N, 0.0f);
        std::vector<float> h_ref(N, 0.0f);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
        cpu_softmax(h_input, h_ref);
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            max_abs_err = std::max(max_abs_err, std::fabs(h_output[i] - h_ref[i]));
            sum += h_output[i];
        }
        std::printf("sum(output)=%.9f, max_abs_err=%.9g\n", sum, max_abs_err);
    } else {
        std::printf("validation=skipped\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(scratch_esum_a));
    CUDA_CHECK(cudaFree(scratch_mx_a));
    CUDA_CHECK(cudaFree(scratch_esum_b));
    CUDA_CHECK(cudaFree(scratch_mx_b));

    return (!validate || max_abs_err < 1e-5f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

# 实验

## 实验设置

由于我比较在意

1. BLOCK STRIDE
1. BLOCK NUMBER

因此我分别选择 BLOCK STRIDE = 1,8,16,32，BLOCK NUMBER(即GridX)有无上限(上限选择sm count * 4)来讨论在N(数据总量)=1e9时kernel的性能。

同时，对于pass1，由于`add_softmax_value`有一串互相依赖的语句看起来无法使用ILP(Instruction-Level Parallelism)优化，因此我也对`add_softmax_value`上面的 `#pragma unroll`的存在是否对性能有影响做了测试

```cpp
//互相依赖的语句
float new_mx = fmaxf(mx, val);
esum = esum * __expf(mx - new_mx) + __expf(val - new_mx);
mx = new_mx;
```

一些默认参数设置：

## 实验结果

### **pass1**

| setting | ms/iter | SMA % | SMI % | Warps % | DRAM R % | DRAM W % | regs/thread | GridX | BlockX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass1_cap_sm4_s1_no_unroll | 3.853 | 95.8 | 32.2 | 38.9 | 27.2 | 1.0 | 25 | 312 | 256 |
| pass1_cap_sm4_s8_unroll | 8.746 | 95.0 | 18.5 | 36.1 | 13.0 | 0.9 | 22 | 312 | 256 |
| pass1_cap_sm4_s8_no_unroll | 9.215 | 94.8 | 21.7 | 35.9 | 12.4 | 0.9 | 20 | 312 | 256 |
| pass1_cap_sm4_s16_unroll | 8.692 | 94.7 | 18.1 | 36.2 | 13.1 | 0.8 | 20 | 312 | 256 |
| pass1_cap_sm4_s16_no_unroll | 9.431 | 94.7 | 20.8 | 35.3 | 12.2 | 0.9 | 20 | 312 | 256 |
| pass1_cap_sm4_s32_unroll | 8.591 | 95.0 | 18.3 | 36.5 | 13.2 | 0.9 | 20 | 312 | 256 |
| pass1_cap_sm4_s32_no_unroll | 7.959 | 95.6 | 24.0 | 38.8 | 13.6 | 0.9 | 20 | 312 | 256 |
| pass1_cap_0_s1_no_unroll | 26.878 | 95.0 | 49.0 | 51.3 | 5.8 | 1.3 | 25 | 3906250 | 256 |
| pass1_cap_0_s8_unroll | 5.527 | 95.5 | 50.1 | 69.0 | 19.0 | 1.2 | 22 | 488282 | 256 |
| pass1_cap_0_s8_no_unroll | 7.530 | 94.7 | 42.2 | 57.3 | 15.2 | 1.3 | 20 | 488282 | 256 |
| pass1_cap_0_s16_unroll | 5.576 | 95.9 | 38.9 | 63.3 | 19.6 | 1.3 | 20 | 244141 | 256 |
| pass1_cap_0_s16_no_unroll | 5.277 | 96.5 | 47.1 | 68.6 | 20.1 | 1.3 | 20 | 244141 | 256 |
| pass1_cap_0_s32_unroll | 5.522 | 95.4 | 33.4 | 61.3 | 19.6 | 1.4 | 20 | 122071 | 256 |
| pass1_cap_0_s32_no_unroll | 5.696 | 94.0 | 38.3 | 61.8 | 19.2 | 1.4 | 20 | 122071 | 256 |

表格展示了所有pass1的抽样数据，其中pass1_cap_*的*代表有无GridX上限，0代表着没有上限，sm4代表上限是sm count * 4, pass1_cap_0_s*_...中的*代表的数字就是BLOCK STRIDE；ms/iter表示每次运行时间；SMA = SM Activity；SMI = SM Issue；Warps = Warps in the flight；DRAM R = DRAR READ; DRAM W = DRAM WRITE；regs/thread表示每个thread占用的寄存器数量；GridX表示每个Grid的block数量；BlockX表示每个block的thread数量

> SMA: 在采样周期内，至少有一个 warp 正在该 SM 上执行/发射指令的时间占比。  
> 即， $\frac{\text{SM active cycles}}{\text{elapsed cycles}}$  
> 其中，active cycle指 该 SM 上存在正在执行的 warp，scheduler 有工作可做，SM 没有完全 idle

> SMI：SM instruction issue slot 被真正用于发射 instruction 的周期占比。  
> 即， $\frac{\text{cycles with issued instructions}}{\text{total cycles}}$

> warps%： 当前 SM 上 resident warps 相对于理论最大 warps 的占比。

以下是对数据的一些解释

Q：为什么cap=0比cap=sm4的SMI大很多？

A： 没有cap时，GridX变得极多，系统里有海量independent warps，pass1中有如下互相依赖的语句

```cpp
float new_mx = fmaxf(mx, val);
esum = esum * __expf(mx - new_mx) + __expf(val - new_mx);
mx = new_mx;
```

很可能出现一个warp在等__expf结果时，scheduler可以立刻找到空闲的warp做事，大大提升latency hiding。也正因如此，cap=0总体比cap=sm4快很多

Q： 为什么pass1_cap_0_s1_no_unroll这么慢？

pass1_cap_0_s1_no_unroll的GridX过于巨大，block scheduling和launch overhead开始主导，反而开始变慢。

其实，H20只可以驻留 SM_COUNT * max_warps_per_sm = 78 * 64 = 4992个warps，SM_COUNT * max_blocks_per_sm = 2496个blocks。cap = 0时的Gridx都远超过GPU simultaneously resident的能力，说明有大量的pending blocks。

这时，BLOCK STRIDE越小，warp做事越少，block生命周期越短，block 结束后，SM 需要不断 dispatch 新的 block 进入 SM，并重新创建 resident warps。由于 s1 中每个 block 工作量极小，block turnover 会显著增加。

虽然大量 independent warps 有助于 latency hiding，但这种收益并不是无限的。当 resident warp pool 已经足够填满 scheduler 后，继续增大 GridX 并不会进一步提升 issue efficiency；相反，block dispatch、block retirement 和 scheduling overhead 会逐渐占据更高比例，最终开始主导 runtime，导致性能劣化。

这一点也可以被 pass1_cap_0_s1_no_unroll 的 Warps% 低于其他 cap=0 setting 所佐证：虽然所有 cap=0 setting 的 GridX 都远超 resident warp 上限，但 s1 中每个 block 生命周期更短，resident warp pool 更难长期维持稳定，因此平均 Warps in Flight % 反而更低。

Q：为什么unroll优化效果不明显

原因是pass1中相互依赖的语句无法受益于ILP，unroll只是把loop iteration展开让编译器更好优化，但有依赖的话总是无法ILP优化。

cap=0时，pending block太多，cap=sm4时，block数量未达上限。  
在BLOCK STRIDE=32是，N/256/32=122070，122070 / SM_COUNT = 1565，因此如果我们继续测试sm8，sm16，sm32，sm64，都会触碰到cap。

由于max_blocks_per_sm=32， max_threads_per_sm=64，BlockX=256，SM能够驻留的block的最大值为min(SM_COUNT*max_threads_per_sm/256, SM_COUNT*max_blocks_per_sm) = 9，大于之前实验中设置的4。可以想象随着cap的增大，性能一定是先优化再劣化。

| setting | ms/iter | SMA % | SMI % | Warps % | DRAM R % | DRAM W % | regs/thread | GridX | BlockX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass1_cap_sm8_s32_no_unroll | 5.343 | 93.3 | 35.6 | 61.2 | 20.1 | 1.1 | 20 | 624 | 256 |
| pass1_cap_sm16_s32_no_unroll | 5.002 | 95.5 | 37.9 | 64.6 | 21.4 | 1.0 | 20 | 1248 | 256 |
| pass1_cap_sm32_s32_no_unroll | 4.752 | 95.8 | 39.8 | 67.3 | 22.4 | 1.1 | 20 | 2496 | 256 |
| pass1_cap_sm64_s32_no_unroll | 4.500 | 95.4 | 41.8 | 70.0 | 23.2 | 1.2 | 20 | 4992 | 256 |

符合预期，性能随着cap的变大而变好，warps%上升，由于这轮实验的cap设置不算大，因此还没有到性能下降的地步。

### pass2

| setting | ms/iter | SMA % | SMI % | Warps % | DRAM R % | DRAM W % | regs/thread | GridX | BlockX | calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass2_cap_sm4_s1 | 0.279 | 90.2 | 8.8 | 29.5 | 8.9 | 4.2 | 28 | 1/2/312 | 256 | 30 |
| pass2_cap_sm4_s8 | 0.054 | 90.0 | 26.0 | 71.8 | 21.4 | 10.8 | 20 | 1/312 | 256 | 20 |
| pass2_cap_sm4_s16 | 0.301 | 89.0 | 8.5 | 29.0 | 8.4 | 3.9 | 20 | 1/312 | 256 | 20 |
| pass2_cap_sm4_s32 | 0.062 | 92.1 | 26.4 | 74.6 | 21.7 | 12.7 | 20 | 1/312 | 256 | 20 |
| pass2_cap_0_s1 | 0.314 | 91.0 | 14.2 | 32.6 | 8.3 | 3.6 | 28 | 1/60/15259 | 256 | 30 |
| pass2_cap_0_s8 | 0.041 | 87.0 | 23.0 | 81.0 | 15.0 | 10.0 | 20 | 1/1908 | 256 | 20 |
| pass2_cap_0_s16 | 0.038 | 92.0 | 30.2 | 87.0 | 24.5 | 18.0 | 20 | 1/954 | 256 | 20 |
| pass2_cap_0_s32 | 0.036 | 92.5 | 30.0 | 84.5 | 25.0 | 13.5 | 20 | 1/477 | 256 | 20 |

### pass3

| setting | ms/iter | SMA % | SMI % | Warps % | DRAM R % | DRAM W % | regs/thread | GridX | BlockX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass3_cap_sm4_s1 | 9.458 | 95.7 | 12.0 | 38.7 | 11.6 | 10.7 | 25 | 312 | 256 |
| pass3_cap_sm4_s8 | 12.289 | 93.9 | 11.0 | 33.3 | 9.9 | 8.6 | 26 | 312 | 256 |
| pass3_cap_sm4_s16 | 11.222 | 94.5 | 11.7 | 35.0 | 10.4 | 9.3 | 28 | 312 | 256 |
| pass3_cap_sm4_s32 | 12.422 | 94.1 | 10.8 | 33.1 | 9.8 | 8.5 | 28 | 312 | 256 |
| pass3_cap_0_s1 | 13.971 | 94.4 | 30.6 | 50.2 | 9.2 | 7.9 | 25 | 3906250 | 256 |
| pass3_cap_0_s8 | 7.035 | 95.1 | 21.0 | 59.5 | 15.7 | 14.6 | 26 | 488282 | 256 |
| pass3_cap_0_s16 | 6.829 | 94.4 | 19.8 | 60.6 | 16.1 | 15.0 | 28 | 244141 | 256 |
| pass3_cap_0_s32 | 7.695 | 94.0 | 17.1 | 56.0 | 14.7 | 13.4 | 28 | 122071 | 256 |
