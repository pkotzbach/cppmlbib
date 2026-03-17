#include "cuda_ops.hpp"
#include "cuda/cuda_kernels.h"
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t status = x; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %d at %s:%d\n", \
               status, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// TODO: copied code
namespace cuda {

std::vector<float> matmul(const std::vector<float>& matrix_A, const std::vector<float>& matrix_B, int K, int X, int Y) {
        float* d_matrix_A;
        float* d_matrix_B;
        float* d_output;
        std::vector<float> output(X*Y);

        size_t matrix_A_bytes = Y * K * sizeof(float);
        size_t matrix_B_bytes = X * K * sizeof(float);
        size_t output_bytes = Y * X * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_matrix_A, matrix_A_bytes));
        CUDA_CHECK(cudaMalloc(&d_matrix_B, matrix_B_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
        CUDA_CHECK(cudaMemcpy(d_matrix_A, matrix_A.data(), matrix_A_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matrix_B, matrix_B.data(), matrix_B_bytes, cudaMemcpyHostToDevice));

        // TODO: fix it
        if (K % 4 == 0 && X % 4 == 0 && X >= 64 && Y >= 64) {
            launch_matmul(d_matrix_A, d_matrix_B, d_output, K, X, Y);
        } else {
            launch_matmul_naive(d_matrix_A, d_matrix_B, d_output, K, X, Y);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_matrix_A));
        CUDA_CHECK(cudaFree(d_matrix_B));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

std::vector<float> matmul_wmma(const std::vector<float>& matrix_A, const std::vector<float>& matrix_B, int K, int X, int Y) {
        float* d_matrix_A;
        float* d_matrix_B;
        float* d_output;
        std::vector<float> output(X*Y);

        size_t matrix_A_bytes = Y * K * sizeof(float);
        size_t matrix_B_bytes = X * K * sizeof(float);
        size_t output_bytes = Y * X * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_matrix_A, matrix_A_bytes));
        CUDA_CHECK(cudaMalloc(&d_matrix_B, matrix_B_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
        CUDA_CHECK(cudaMemcpy(d_matrix_A, matrix_A.data(), matrix_A_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matrix_B, matrix_B.data(), matrix_B_bytes, cudaMemcpyHostToDevice));

        launch_matmul_wmma(d_matrix_A, d_matrix_B, d_output, K, X, Y);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_matrix_A));
        CUDA_CHECK(cudaFree(d_matrix_B));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

std::vector<float> matmul_naive(const std::vector<float>& matrix_A, const std::vector<float>& matrix_B, int K, int X, int Y) {
        float* d_matrix_A;
        float* d_matrix_B;
        float* d_output;
        std::vector<float> output(X*Y);

        size_t matrix_A_bytes = Y * K * sizeof(float);
        size_t matrix_B_bytes = X * K * sizeof(float);
        size_t output_bytes = Y * X * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_matrix_A, matrix_A_bytes));
        CUDA_CHECK(cudaMalloc(&d_matrix_B, matrix_B_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
        CUDA_CHECK(cudaMemcpy(d_matrix_A, matrix_A.data(), matrix_A_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matrix_B, matrix_B.data(), matrix_B_bytes, cudaMemcpyHostToDevice));

        launch_matmul_naive(d_matrix_A, d_matrix_B, d_output, K, X, Y);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_matrix_A));
        CUDA_CHECK(cudaFree(d_matrix_B));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

std::vector<float> matmul_cublas(const std::vector<float>& matrix_A, const std::vector<float>& matrix_B, int K, int X, int Y) {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
                CUBLAS_CHECK(cublasCreate(&handle));
                CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        }

        float* d_matrix_A;
        float* d_matrix_B;
        float* d_output;
        std::vector<float> output(X*Y);

        size_t matrix_A_bytes = Y * K * sizeof(float);
        size_t matrix_B_bytes = X * K * sizeof(float);
        size_t output_bytes = Y * X * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_matrix_A, matrix_A_bytes));
        CUDA_CHECK(cudaMalloc(&d_matrix_B, matrix_B_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
        CUDA_CHECK(cudaMemcpy(d_matrix_A, matrix_A.data(), matrix_A_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matrix_B, matrix_B.data(), matrix_B_bytes, cudaMemcpyHostToDevice));

        float alpha = 1.0f;
        float beta = 0.0f;

        // C = alpha * A * B + beta * C
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, X, Y, K, &alpha, d_matrix_B, X, d_matrix_A, K, &beta, d_output, X));

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_matrix_A));
        CUDA_CHECK(cudaFree(d_matrix_B));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

std::vector<float> binary_op(const char op, const std::vector<float>& matrix_A, const std::vector<float>& matrix_B, int size) {
        float* d_matrix_A;
        float* d_matrix_B;
        float* d_output;
        std::vector<float> output(size);

        size_t size_bytes = sizeof(float) * size;

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

void softmax(const std::vector<float>& input, float* output, int N, int C) {
        if (C > 1024) {
                // TODO: fix this - its because max thread block size is 1024
                throw std::invalid_argument("CUDA softmax currently implemented for C <= 1024");
        }
        float* d_input;
        float* d_output;
        int size = N*C;

        size_t size_bytes = sizeof(float) * size;

        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), size_bytes, cudaMemcpyHostToDevice));

        launch_softmax2(d_input, d_output, N, C);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output, d_output, size_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
}

float reduction(const ReductionOp op, const std::span<const float>& input) {
        float* d_input;
        float* d_output;
        int size = input.size();
        
        size_t size_bytes = sizeof(float) * size;
        
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), size_bytes, cudaMemcpyHostToDevice));
        
        // TODO: while here or in launch_reduction?
        while(size > 1) {
                size = launch_reduction(op, d_input, d_output, size);
                CUDA_CHECK(cudaMemcpy(d_input, d_output, sizeof(float) * size, cudaMemcpyDeviceToDevice));
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float output;
        CUDA_CHECK(cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

float full_reduction(const ReductionOp op, const std::span<const float>& input) {
        float* d_input;
        float* d_output;
        int size = input.size();
        
        size_t size_bytes = sizeof(float) * size;
        
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), size_bytes, cudaMemcpyHostToDevice));
        
        launch_full_reduction(op, d_input, d_output, size);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float output;
        CUDA_CHECK(cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

} // namespace cuda