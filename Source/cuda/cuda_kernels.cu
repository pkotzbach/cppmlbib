#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

#ifdef CUDA_TEST
#include "cuda_debug.h"
int g_cuda_matmul_launches = 0;
#endif

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
    g_cuda_matmul_launches++;
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

__global__ void softmax_kernel(const double* input, double* output, int N, int C) {

}

void launch_softmax(const double* input, double* output, int N, int C) {
}
