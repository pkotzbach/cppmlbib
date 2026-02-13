#include "matmul.hpp"
#include "cuda_matmul.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

std::vector<double> naive_matmul(const std::vector<double> &A, const std::vector<double> &B, int K, int X, int Y)
{
    std::vector<double> output(X * Y, 0.0);

    for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
            for (int k = 0; k < K; ++k)
                output[y * X + x] += A[y * K + k] * B[k * X + x];

    return output;
}

std::vector<double> cuda_matmul(const std::vector<double>& matrix_A, const std::vector<double>& matrix_B, int K, int X, int Y)
{
        double *d_matrix_A, *d_matrix_B, *d_output;
        std::vector<double> output(X*Y);

        size_t matrix_A_bytes = Y * K * sizeof(double);
        size_t matrix_B_bytes = X * K * sizeof(double);
        size_t output_bytes = Y * X * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_matrix_A, matrix_A_bytes));
        CUDA_CHECK(cudaMalloc(&d_matrix_B, matrix_B_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
        CUDA_CHECK(cudaMemcpy(d_matrix_A, matrix_A.data(), matrix_A_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matrix_B, matrix_B.data(), matrix_B_bytes, cudaMemcpyHostToDevice));

        launch_matmul(
            d_matrix_A, d_matrix_B, d_output, K, X, Y
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

        // for (int i = 0; i < Y; i++) {
        //     for (int j = 0; j < X; j++) {
        //         printf("output[%d,%d] = %f\n", i, j, output[i * X + j]);
        //     }
        // }

        CUDA_CHECK(cudaFree(d_matrix_A));
        CUDA_CHECK(cudaFree(d_matrix_B));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

std::vector<double> _matmul(const std::vector<double> A, const std::vector<double> B, int K, int X, int Y, const std::string device)
{
    if (device == "cpu")
        return naive_matmul(A, B, K, X, Y);
    else if (device == "cuda")
        return cuda_matmul(A, B, K, X, Y);
    return {};
}
