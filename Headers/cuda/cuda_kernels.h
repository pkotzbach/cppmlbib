#pragma once
#include "cuda_ops.hpp"
#include "globals.hpp"

void launch_matmul(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y);
void launch_matmul_wmma(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y);
void launch_matmul_naive(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y);

void launch_binary_op(const char op, const float* input_A, const float* input_B, float* output, int size);
void launch_softmax2(const float* input_A, float* output, int N, int C);
int launch_reduction(const ReductionOp op, const float* input, float* output, int size);
void launch_full_reduction(const ReductionOp op, const float* input, float* output, int size);