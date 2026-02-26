#pragma once

void launch_matmul(const double* d_A, const double* d_B, double* d_C, int K, int X, int Y);
void launch_softmax(const double* input, double* output, int N, int C);