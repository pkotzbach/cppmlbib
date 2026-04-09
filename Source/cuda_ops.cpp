#include "cuda_ops.hpp"
#include "cuda/cuda_kernels.h"
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) [[unlikely]] { \
        printf("CUDA error %s at %s:%d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t status = x; \
    if (status != CUBLAS_STATUS_SUCCESS) [[unlikely]] { \
        printf("cuBLAS error %d at %s:%d\n", \
               status, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// TODO: copied code
namespace cuda {

void matmul(const float* matrix_A, const float* matrix_B, float* output, int K, int X, int Y) {
        // TODO: shouldnt be hardcoded
        if (K % 4 == 0 && X % 4 == 0 && X >= 64 && Y >= 64 && K >= 8) {
            launch_matmul(matrix_A, matrix_B, output, K, X, Y);
        } else {
            launch_matmul_nonvec(matrix_A, matrix_B, output, K, X, Y);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
}

void matmul_tc(const float* matrix_A, const float* matrix_B, float* output, int K, int X, int Y) {
        launch_matmul_tc(matrix_A, matrix_B, output, K, X, Y);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
}

void matmul_naive(const float* matrix_A, const float* matrix_B, float* output, int K, int X, int Y) {
        launch_matmul_naive(matrix_A, matrix_B, output, K, X, Y);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
}

void matmul_cublas(const float* matrix_A, const float* matrix_B, float* output, int K, int X, int Y) {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
                CUBLAS_CHECK(cublasCreate(&handle));
                CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        }

        float alpha = 1.0f;
        float beta = 0.0f;

        // C = alpha * A * B + beta * C
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, X, Y, K, &alpha, matrix_B, X, matrix_A, K, &beta, output, X));

        CUDA_CHECK(cudaDeviceSynchronize());
}

void binary_op(const char op, const float* matrix_A, const float* matrix_B, float* output, int size) {
        launch_binary_op(op, matrix_A, matrix_B, output, size);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
}

void binary_op_strided(const char op, const float* d_matrix_A, std::array<int, MAX_DIMS> strides_A, 
                                                    const float* d_matrix_B, std::array<int, MAX_DIMS> strides_B, 
                                                    std::array<int, MAX_DIMS> shape, int size, int dims, float* d_output) {
                                                        
        launch_binary_op_strided(op, d_matrix_A, strides_A, d_matrix_B, strides_B, shape, d_output, size, dims);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
}

void binary_op_backward_strided(const char op, const float* input_A, std::array<int, MAX_DIMS> strides_A,
                                   const float* input_B, std::array<int, MAX_DIMS> strides_B,
                                   float* grad_A, float* grad_B, const float* grad_output,
                                   std::array<int, MAX_DIMS> shape, int size, int dims) {
    launch_binary_op_backward_strided(op, input_A, strides_A, input_B, strides_B, grad_A, grad_B, grad_output, shape, size, dims);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void softmax(const float* d_input, float* d_output, int N, int C) {
        if (C > 1024) {
                // TODO: fix this - its because max thread block size is 1024
                throw std::invalid_argument("CUDA softmax currently implemented for C <= 1024");
        }
        launch_softmax2(d_input, d_output, N, C);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
}

float reduction(const ReductionOp op, const float* input, int size) {
        float* d_input;
        float* d_output;
        
        size_t size_bytes = sizeof(float) * size;
        
        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input, size_bytes, cudaMemcpyDeviceToDevice));
        
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

float full_reduction(const ReductionOp op, const float* input, int size) {
        float* d_output;
        
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(float))); // only one result
        
        launch_full_reduction(op, input, d_output, size);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float output;
        CUDA_CHECK(cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_output));
        
        return output;
}

// it breaks strides
void make_continous(Tensor_ptr tensor)
{
        float* values_ptr = tensor->raw_values();
        float* grads_ptr = tensor->raw_grads();
        int count = tensor->get_total_count();
        float* d_output;
        int buffer_size = sizeof(float) * count;
        cudaMalloc(&d_output, buffer_size);

        // values
        launch_make_continous(values_ptr, d_output, count, tensor->get_strides(), tensor->get_shape());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaMemcpy(values_ptr, d_output, buffer_size, cudaMemcpyDeviceToDevice);

        // grads
        launch_make_continous(grads_ptr, d_output, count, tensor->get_strides(), tensor->get_shape());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaMemcpy(grads_ptr, d_output, buffer_size, cudaMemcpyDeviceToDevice);

        cudaFree(d_output);

        tensor->set_strides(stride::calc_strides(tensor->get_shape()));
}

void im2col(const float *in_data, float *res_data, int batch, int height, int width, int out_h, int out_w, int channels, int kernel_size, int stride, int padding)
{
    launch_im2col(in_data, res_data, batch, height, width, out_h, out_w, channels, kernel_size, stride, padding);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void relu(const float* input, float* output, int size) {
    launch_relu(input, output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void exp(const float* input, float* output, int size) {
    launch_exp(input, output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void relu_backward(const float* input, float* grad_input, const float* grad_output, int size) {
    launch_relu_backward(input, grad_input, grad_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sum_backward(float* grad_input, const float* grad_output, int size) {
    launch_sum_backward(grad_input, grad_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sum_axis_backward(float* grad_input, const float* grad_output, int N, int C, int axis) {
    launch_sum_axis_backward(grad_input, grad_output, N, C, axis);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void exp_backward(const float* output, float* grad_input, const float* grad_output, int size) {
    launch_exp_backward(output, grad_input, grad_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void matmul_backward(const float* A, const float* B, float* grad_A, float* grad_B, const float* grad_output, int K, int X, int Y) {
    // grad_A (Y, K) = grad_output (Y, X) * B^T (X, K)
    // grad_B (K, X) = A^T (K, Y) * grad_output (Y, X)

    float *d_BT, *d_AT;
    CUDA_CHECK(cudaMalloc(&d_BT, X * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AT, K * Y * sizeof(float)));

    std::vector<int> shape_A = {Y, K};
    std::vector<int> strides_A = {K, 1};
    std::vector<int> shape_AT = {K, Y};
    std::vector<int> strides_AT = {1, K};

    std::vector<int> shape_B = {K, X};
    std::vector<int> strides_B = {X, 1};
    std::vector<int> shape_BT = {X, K};
    std::vector<int> strides_BT = {1, X};

    launch_make_continous(A, d_AT, K * Y, strides_AT, shape_AT);
    launch_make_continous(B, d_BT, X * K, strides_BT, shape_BT);

    float *d_grad_A_temp, *d_grad_B_temp;
    CUDA_CHECK(cudaMalloc(&d_grad_A_temp, Y * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_B_temp, K * X * sizeof(float)));

    launch_matmul_naive(grad_output, d_BT, d_grad_A_temp, X, K, Y);
    launch_matmul_naive(d_AT, grad_output, d_grad_B_temp, Y, X, K);

    launch_binary_op('+', grad_A, d_grad_A_temp, grad_A, Y * K);
    launch_binary_op('+', grad_B, d_grad_B_temp, grad_B, K * X);

    CUDA_CHECK(cudaFree(d_BT));
    CUDA_CHECK(cudaFree(d_AT));
    CUDA_CHECK(cudaFree(d_grad_A_temp));
    CUDA_CHECK(cudaFree(d_grad_B_temp));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void softmax_backward(const float* output, float* grad_input, const float* grad_output, int N, int C) {
    launch_softmax_backward(output, grad_input, grad_output, N, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void transpose(float* matrix, float* matrixT, int N, int C) {
    launch_transpose(matrix, matrixT, N, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sgd_step(float* values, const float* grads, float lr, int total_count) {
    launch_sgd_step(values, grads, lr, total_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda