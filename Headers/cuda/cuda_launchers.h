#pragma once

void launch_matmul(const double* d_A, const double* d_B, double* d_C, int K, int X, int Y);
void launch_binary_op(const char op, const double* input_A, const double* input_B, double* output, int size);