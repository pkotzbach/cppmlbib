#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <cuda/cmath>

#ifdef CUDA_TEST
#include "cuda_debug.h"
int g_cuda_kernel_launches = 0;
#endif

template <char Op>
__global__ void binary_op_kernel(const double* input_A, const double* input_B, double* output, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
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

__global__ void softmax_kernel(const double* input, double* output, int N, int C) {

}

void launch_softmax(const double* input, double* output, int N, int C) {
}