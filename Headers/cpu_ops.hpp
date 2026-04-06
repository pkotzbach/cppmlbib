#pragma once
#include <vector>
#include <span>
#include "globals.hpp"

namespace cpu {
    void matmul(const float* A_ptr, const float* B_ptr, float* C_ptr, int K, int X, int Y);
    void BT_matmul(const float* A_ptr, const float* B_ptr, float* C_ptr, int K, int M, int N);
    void AT_matmul(const float* A_ptr, const float* B_ptr, float* C_ptr, int K, int M, int N);
    void matmul_naive(const float* A_ptr, const float* B_ptr, float* C_ptr, int K, int X, int Y);

    void softmax(const float* __restrict__ input, float* __restrict__ output, int N, int C);
}