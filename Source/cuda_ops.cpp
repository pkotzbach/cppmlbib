#include "cuda_ops.hpp"
#include "cuda/cuda_launchers.h"
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// TODO: copied code
namespace cuda {

std::vector<double> matmul(const std::vector<double>& matrix_A, const std::vector<double>& matrix_B, int K, int X, int Y)
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

        launch_matmul(d_matrix_A, d_matrix_B, d_output, K, X, Y);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_matrix_A));
        CUDA_CHECK(cudaFree(d_matrix_B));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

// TODO: as template with "op"? then i need to store it in .hpp and i dont know if i want that
std::vector<double> binary_op(const char op, const std::vector<double>& matrix_A, const std::vector<double>& matrix_B, int size)
{
        double *d_matrix_A, *d_matrix_B, *d_output;
        std::vector<double> output(size);

        size_t size_bytes = sizeof(double) * size;

        CUDA_CHECK(cudaMalloc(&d_matrix_A, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_matrix_B, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));
        CUDA_CHECK(cudaMemcpy(d_matrix_A, matrix_A.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matrix_B, matrix_B.data(), size_bytes, cudaMemcpyHostToDevice));

        launch_binary_op(op, d_matrix_A, d_matrix_B, d_output, size);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_matrix_A));
        CUDA_CHECK(cudaFree(d_matrix_B));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

std::vector<double> softmax(const std::vector<double>& matrix_A, int N, int C)
{
        double *d_matrix_A, *d_output;
        int size = N*C;
        std::vector<double> output(size);

        size_t size_bytes = sizeof(double) * size;

        CUDA_CHECK(cudaMalloc(&d_matrix_A, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));
        CUDA_CHECK(cudaMemcpy(d_matrix_A, matrix_A.data(), size_bytes, cudaMemcpyHostToDevice));

        launch_softmax(d_matrix_A, d_output, N, C);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_matrix_A));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

double reduction(const ReductionOp op, const std::span<const double>& input)
{
        double *d_input, *d_output;
        int size = input.size();
        
        size_t size_bytes = sizeof(double) * size;
        
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), size_bytes, cudaMemcpyHostToDevice));
        
        while(size > 1) {
                size = launch_reduction(op, d_input, d_output, size);
                CUDA_CHECK(cudaMemcpy(d_input, d_output, sizeof(double) * size, cudaMemcpyDeviceToDevice));
        }
        
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double output;
        CUDA_CHECK(cudaMemcpy(&output, d_output, sizeof(double), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

} // namespace cuda