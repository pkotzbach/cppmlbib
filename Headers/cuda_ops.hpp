#pragma once
#include <vector>
#include <span>
#include "globals.hpp"
#include "tensor.hpp"

namespace cuda {
    void matmul(const float* matrix_A, const float* matrix_B, float* output, int K, int X, int Y);
    void matmul_tc(const float* matrix_A, const float* matrix_B, float* output, int K, int X, int Y);
    void matmul_naive(const float* matrix_A, const float* matrix_B, float* output, int K, int X, int Y);
    void matmul_cublas(const float* matrix_A, const float* matrix_B, float* output, int K, int X, int Y);
    void binary_op(const char op, const float* matrix_A, const float* matrix_B, float* output, int size);
    void binary_op_strided(const char op, const float* input_A, std::array<int, MAX_DIMS> strides_A,
                              const float* input_B, std::array<int, MAX_DIMS> strides_B, std::array<int, MAX_DIMS> shape,
                              int size, int dims, float* output);
    void binary_op_backward_strided(const char op, const float* input_A, std::array<int, MAX_DIMS> strides_A,
                                   const float* input_B, std::array<int, MAX_DIMS> strides_B,
                                   float* grad_A, float* grad_B, const float* grad_output,
                                   std::array<int, MAX_DIMS> shape, int size, int dims);
    void softmax(const float* input, float* output, int N, int C);

    void transpose(float* matrix, float* matrixT, int N, int C);

    float reduction(const ReductionOp op, const float* input, int size);
    float full_reduction(const ReductionOp op, const float* input, int size);

    void make_continous(Tensor_ptr tensor);
    void im2col(const float* in_data, float* res_data, int batch, int height, int width, int out_h, int out_w, 
                int channels, int kernel_size, int stride, int padding);

    void relu(const float* input, float* output, int size);
    void exp(const float* input, float* output, int size);

    // Backward operations
    void relu_backward(const float* input, float* grad_input, const float* grad_output, int size);
    void sum_backward(float* grad_input, const float* grad_output, int size);
    void sum_axis_backward(float* grad_input, const float* grad_output, int N, int C, int axis);
    void exp_backward(const float* output, float* grad_input, const float* grad_output, int size);
    void matmul_backward(const float* A, const float* B, float* grad_A, float* grad_B, const float* grad_output, int K, int X, int Y);
    void softmax_backward(const float* output, float* grad_input, const float* grad_output, int N, int C);
}