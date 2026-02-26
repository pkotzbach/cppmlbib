#pragma once
#include <vector>

namespace cpu {
    std::vector<double> matmul(const std::vector<double> &A, const std::vector<double> &B, int K, int X, int Y);
    std::vector<double> softmax(const std::vector<double>& input, int N, int C);
}