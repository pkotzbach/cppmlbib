#pragma once
#include <vector>

namespace cuda {
    std::vector<double> matmul(const std::vector<double>& matrix_A, const std::vector<double>& matrix_B, int K, int X, int Y);
    std::vector<double> binary_op(const char op, const std::vector<double>& matrix_A, const std::vector<double>& matrix_B, int size);
    std::vector<double> softmax(const std::vector<double>& matrix_A, int N, int C);
}