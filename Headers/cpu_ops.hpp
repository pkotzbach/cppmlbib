#pragma once
#include <vector>
#include <span>
#include "globals.hpp"

namespace cpu {
    std::vector<double> matmul(const std::vector<double> &A, const std::vector<double> &B, int K, int X, int Y);
    void softmax(const std::vector<double>& input, double* output, int N, int C);
}