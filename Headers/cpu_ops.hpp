#pragma once
#include <vector>
#include <span>
#include "globals.hpp"

namespace cpu {
    std::vector<float> matmul(const std::vector<float> &A, const std::vector<float> &B, int K, int X, int Y);
    void softmax(const std::vector<float>& input, float* output, int N, int C);
}