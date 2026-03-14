#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <cuda/cmath>
#include "cuda_ops.hpp"

#ifdef CUDA_TEST
#include "cuda_debug.h"
int g_cuda_kernel_launches = 0;
#endif

template <char Op>
__global__ void binary_op_kernel(const float* input_A, const float* input_B, float* output, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        if constexpr (Op == '+') output[idx] = input_A[idx] + input_B[idx];
        else if constexpr (Op == '-') output[idx] = input_A[idx] - input_B[idx];
        else if constexpr (Op == '*') output[idx] = input_A[idx] * input_B[idx];
        else if constexpr (Op == '/') output[idx] = input_A[idx] / input_B[idx];
    }
}

void launch_binary_op(const char op, const float* input_A, const float* input_B, float* output, int size) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    switch (op) {
        case '+': binary_op_kernel<'+'><<<blocks, threads>>>(input_A, input_B, output, size); break;
        case '-': binary_op_kernel<'-'><<<blocks, threads>>>(input_A, input_B, output, size); break;
        case '*': binary_op_kernel<'*'><<<blocks, threads>>>(input_A, input_B, output, size); break;
        case '/': binary_op_kernel<'/'><<<blocks, threads>>>(input_A, input_B, output, size); break;
        default: throw std::invalid_argument("Unknown op");
    }
}

// -------------------

__global__ void matmul_kernel_naive(const float* A, const float* B, float* C, int K, int X, int Y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < Y && x < X) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[y * K + k] * B[k * X + x];
        }
        C[y * X + x] = sum;
    }
}

void launch_matmul_naive(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y)
{
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif

    dim3 block(16, 16);
    dim3 grid(cuda::ceil_div(X, block.x), cuda::ceil_div(Y, block.y));

    matmul_kernel_naive<<<grid, block>>>(d_A, d_B, d_C, K, X, Y);
}


#define BLOCK_K 8
#define PER_THREAD 8
#define BLOCK_X 64
#define BLOCK_Y 64
__global__ void matmul_kernel(const float* A, const float* B, float* C, int K, int X, int Y)
{
    __shared__ float sA[BLOCK_Y * BLOCK_K];
    __shared__ float sB[BLOCK_K * BLOCK_X];

    const int block_steps = BLOCK_X / PER_THREAD; // or BLOCK_Y
    
    const int thread_x = threadIdx.x % block_steps;
    const int thread_y = threadIdx.x / block_steps;

    float sum[PER_THREAD * PER_THREAD] = {0};

    float regA[PER_THREAD] = {0};
    float regB[PER_THREAD] = {0};

    for (int k = 0; k < K; k += BLOCK_K) {

        for (int y = 0; y < block_steps; ++y) {
            int offset = y * PER_THREAD;
            sA[(thread_y + offset) * BLOCK_K + thread_x] = A[(blockIdx.y * BLOCK_Y + thread_y + offset) * K + thread_x + k];
            sB[y * BLOCK_X + threadIdx.x] = B[(k + y) * X + blockIdx.x * BLOCK_X + threadIdx.x];
        }

        __syncthreads();

        // inner loop
        for (int k_in = 0; k_in < PER_THREAD; ++k_in) {
            for (int i = 0; i < PER_THREAD; ++i) {
                regA[i] = sA[(thread_y * PER_THREAD + i) * BLOCK_K + k_in];
                regB[i] = sB[k_in * BLOCK_X + thread_x * PER_THREAD + i];
            }

            for (int i = 0; i < PER_THREAD; ++i) {
                for (int j = 0; j < PER_THREAD; ++j) {
                    sum[i * PER_THREAD + j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    for (int y = 0; y < PER_THREAD; ++y)
        for (int x = 0; x < PER_THREAD; ++x) {
            int C_y = blockIdx.y * BLOCK_Y + thread_y * PER_THREAD + y;
            int C_x = blockIdx.x * BLOCK_X + thread_x * PER_THREAD + x;

            C[C_y * X + C_x] = sum[y * PER_THREAD + x];
        }
}

void launch_matmul(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y)
{
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    static_assert(BLOCK_X == BLOCK_Y);
    static_assert(BLOCK_Y / PER_THREAD == PER_THREAD);
    dim3 grid(cuda::ceil_div(X, BLOCK_X), cuda::ceil_div(Y, BLOCK_Y));
    dim3 block(BLOCK_X / PER_THREAD * BLOCK_Y / PER_THREAD);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, K, X, Y);
}

// ----------------------------------

// TODO: could be faster https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <ReductionOp op>
__global__ void reduction_kernel(const float* input, float* output, int size)
{
    extern __shared__ float sharedArray[];

    int bidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidx = threadIdx.x;

    sharedArray[tidx] = bidx < size? input[bidx]: 0;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            if constexpr (op == ReductionOp::MAX) {
                if (sharedArray[tidx] < sharedArray[tidx + i]) sharedArray[tidx] = sharedArray[tidx + i];
            }
            if constexpr (op == ReductionOp::MIN) {
                if (sharedArray[tidx] > sharedArray[tidx + i]) sharedArray[tidx] = sharedArray[tidx + i];
            }
            if constexpr (op == ReductionOp::SUM) {
                sharedArray[tidx] += sharedArray[tidx + i];
            }
        }
        __syncthreads();
    }

    if (tidx == 0) output[blockIdx.x] = sharedArray[0];
}

int launch_reduction(const ReductionOp op, const float* input, float* output, int size) {
#ifdef CUDA_TEST 
    g_cuda_kernel_launches++; 
#endif 

    int block = 256; 
    int grid = cuda::ceil_div(size, block);
    int shared_memory = sizeof(float) * block; 
    switch (op) {
        case MAX: 
            reduction_kernel<MAX><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        case MIN: 
            reduction_kernel<MIN><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        case SUM: 
            reduction_kernel<SUM><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        default: throw std::invalid_argument("Unknown op");
    }

    return grid;
}

// ----------------------------------

template <ReductionOp op>
__global__ void reduction_kernel2(const float* input, float* output, int size)
{
    extern __shared__ float sharedArray[];
    int tidx = threadIdx.x;
    
    float local_max = -DBL_MAX;
    for (int i = tidx; i < size; i += blockDim.x)
        local_max = fmax(local_max, input[i]);

    sharedArray[tidx] = local_max;
    
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if constexpr (op == ReductionOp::MAX) {
            if (tidx < i && sharedArray[tidx] < sharedArray[tidx + i])
                sharedArray[tidx] = sharedArray[tidx + i];
        }
        __syncthreads();
    }

    if (tidx == 0) *output = sharedArray[0];
}

void launch_full_reduction(const ReductionOp op, const float* input, float* output, int size) {
#ifdef CUDA_TEST 
    g_cuda_kernel_launches++; 
#endif 

    int block = 256; 
    int grid = cuda::ceil_div(size, block);
    int shared_memory = sizeof(float) * block; 
    switch (op) {
        case ReductionOp::MAX: 
            reduction_kernel2<ReductionOp::MAX><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        default: throw std::invalid_argument("Unknown op");
    }
}

// -----------------

__global__ void softmax_kernel2(const float* input, float* output, int N, int C) {
    extern __shared__ float sharedArray[];
    int n = blockIdx.x;
    int c = threadIdx.x;
    
    bool active = c < C;
    sharedArray[c] = active? input[n * C + c]: -DBL_MAX;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (c < i && c + i < C && sharedArray[c] < sharedArray[c + i])
            sharedArray[c] = sharedArray[c + i];
        __syncthreads();
    }

    float max_val = sharedArray[0];
    // printf("%d: %f\n", c, max_val);
    __syncthreads();

    float exp_val;
    if (active) {
        exp_val = exp(input[n * C + c] - max_val);
        sharedArray[c] = exp_val;
        // printf("%d: %f\n", c, exp_val);
    }
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (c < i && c + i < C) {
            // printf("%d: %f += %f\n", c, sharedArray[c], sharedArray[c + i]);
            sharedArray[c] += sharedArray[c + i];
        }
        __syncthreads();
    }
    
    if (active) {
        // printf("%d: %f / %f\n", c, exp_val, sharedArray[0]);
        output[n * C + c] = exp_val / sharedArray[0];
    }
}

void launch_softmax2(const float* input, float* output, int N, int C) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    // one row per block
    int block = (C + 32 - 1) / 32 * 32;
    size_t shared_memory_bytes = block * sizeof(float);
    int grid = N;

    softmax_kernel2<<<grid, block, shared_memory_bytes>>>(input, output, N, C);
}