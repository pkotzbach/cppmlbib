#pragma once
#include "cuda_ops.hpp"
#include "globals.hpp"


void launch_matmul(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int K, int X, int Y);
void launch_matmul_tc(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int K, int X, int Y);
void launch_matmul_naive(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int K, int X, int Y);

void launch_binary_op(const char op, const float* input_A, const float* input_B, float* output, int size);
void launch_binary_op_strided(const char op, const float* input_A, std::array<int, MAX_DIMS> strides_A,
                              const float* input_B, std::array<int, MAX_DIMS> strides_B, std::array<int, MAX_DIMS> shape,
                              float* output, int size, int dims);
void launch_binary_op_backward_strided(const char op, const float* input_A, std::array<int, MAX_DIMS> strides_A,
                                      const float* input_B, std::array<int, MAX_DIMS> strides_B,
                                      float* grad_A, float* grad_B, const float* grad_output,
                                      std::array<int, MAX_DIMS> shape, int size, int dims);
void launch_softmax2(const float* input_A, float* output, int N, int C);
int launch_reduction(const ReductionOp op, const float* input, float* output, int size);
void launch_full_reduction(const ReductionOp op, const float* input, float* output, int size);

void launch_transpose(float* matrix, float* matrixT, int N, int C);
void launch_im2col(const float* __restrict__ in_data, float* __restrict__ res_data, int batches, int height, int width, int out_h, int out_w, 
    int channels, int kernel_size, int stride, int padding);

void launch_make_continous(const float* val, float* output, int count, const std::vector<int> &strides, const std::vector<int> &shape);

void launch_relu(const float* input, float* output, int size);
void launch_exp(const float* input, float* output, int size);

void launch_relu_backward(const float* input, float* grad_input, const float* grad_output, int size);
void launch_sum_backward(float* grad_input, const float* grad_output, int size);
void launch_sum_axis_backward(float* grad_input, const float* grad_output, int N, int C, int axis);
void launch_exp_backward(const float* output, float* grad_input, const float* grad_output, int size);
void launch_softmax_backward(const float* output, float* grad_input, const float* grad_output, int N, int C);
