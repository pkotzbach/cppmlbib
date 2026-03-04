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
__global__ void binary_op_kernel(const double* input_A, const double* input_B, double* output, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        if constexpr (Op == '+') output[idx] = input_A[idx] + input_B[idx];
        else if constexpr (Op == '-') output[idx] = input_A[idx] - input_B[idx];
        else if constexpr (Op == '*') output[idx] = input_A[idx] * input_B[idx];
        else if constexpr (Op == '/') output[idx] = input_A[idx] / input_B[idx];
    }
}

void launch_binary_op(const char op, const double* input_A, const double* input_B, double* output, int size) {
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

__global__ void matmul_kernel(
    const double* A,   // [Y, K]
    const double* B,   // [K, X]
    double* C,         // [Y, X]
    int K,
    int X,
    int Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output column

    if (row < Y && col < X) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] *
                   B[k * X + col];
        }
        C[row * X + col] = sum;
    }
}

void launch_matmul(
    const double* d_A,
    const double* d_B,
    double* d_C,
    int K,
    int X,
    int Y)
{
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif

    dim3 block(16, 16);
    dim3 grid(
        (X + block.x - 1) / block.x,
        (Y + block.y - 1) / block.y
    );

    matmul_kernel<<<grid, block>>>(
        d_A, d_B, d_C, K, X, Y
    );
}

// ----------------------------------

// TODO: could be faster https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <ReductionOp op>
__global__ void reduction_kernel(const double* input, double* output, int size)
{
    extern __shared__ double sharedArray[];

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

int launch_reduction(const ReductionOp op, const double* input, double* output, int size) {
#ifdef CUDA_TEST 
    g_cuda_kernel_launches++; 
#endif 

    int block = 256; 
    int grid = cuda::ceil_div(size, block);
    int shared_memory = sizeof(double) * block; 
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
__global__ void reduction_kernel2(const double* input, double* output, int size)
{
    extern __shared__ double sharedArray[];
    int tidx = threadIdx.x;
    
    double local_max = -DBL_MAX;
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

void launch_full_reduction(const ReductionOp op, const double* input, double* output, int size) {
#ifdef CUDA_TEST 
    g_cuda_kernel_launches++; 
#endif 

    int block = 256; 
    int grid = cuda::ceil_div(size, block);
    int shared_memory = sizeof(double) * block; 
    switch (op) {
        case ReductionOp::MAX: 
            reduction_kernel2<ReductionOp::MAX><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        default: throw std::invalid_argument("Unknown op");
    }
}

// -----------------

__global__ void softmax_kernel2(const double* input, double* output, int N, int C) {
    extern __shared__ double sharedArray[];
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

    double max_val = sharedArray[0];
    printf("%d: %f\n", c, max_val);
    __syncthreads();

    double exp_val;
    if (active) {
        exp_val = exp(input[n * C + c] - max_val);
        sharedArray[c] = exp_val;
        printf("%d: %f\n", c, exp_val);
    }
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (c < i && c + i < C) {
            printf("%d: %f += %f\n", c, sharedArray[c], sharedArray[c + i]);
            sharedArray[c] += sharedArray[c + i];
        }
        __syncthreads();
    }
    
    if (active) {
        printf("%d: %f / %f\n", c, exp_val, sharedArray[0]);
        output[n * C + c] = exp_val / sharedArray[0];
    }
}

void launch_softmax2(const double* input, double* output, int N, int C) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    // one row per block
    int block = (C + 32 - 1) / 32 * 32;
    size_t shared_memory_bytes = block * sizeof(double);
    int grid = N;

    softmax_kernel2<<<grid, block, shared_memory_bytes>>>(input, output, N, C);
}