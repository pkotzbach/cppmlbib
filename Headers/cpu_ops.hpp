#pragma once
#include <vector>
#include <span>
#include "globals.hpp"

namespace cpu {
    std::vector<float> matmul(const std::vector<float> &A, const std::vector<float> &B, int K, int X, int Y);
    std::vector<float> matmul(const float* A_ptr, const float* B_ptr, int K, int X, int Y);
    std::vector<float> BT_matmul(const float* A_ptr, const float* B_ptr, int K, int M, int N);
    std::vector<float> matmul_naive(const std::vector<float> &A, const std::vector<float> &B, int K, int X, int Y);

    void softmax(const std::vector<float>& input, float* output, int N, int C);
    void softmax(const float* __restrict__ input, float* __restrict__ output, int N, int C);
}