#pragma once
#include "cuda_ops.hpp"
#include "globals.hpp"
// TODO: rename file to cuda_kernels?

void launch_matmul(const double* d_A, const double* d_B, double* d_C, int K, int X, int Y);
void launch_binary_op(const char op, const double* input_A, const double* input_B, double* output, int size);
void launch_softmax2(const double* input_A, double* output, int N, int C);
int launch_reduction(const ReductionOp op, const double* input, double* output, int size);
void launch_full_reduction(const ReductionOp op, const double* input, double* output, int size);