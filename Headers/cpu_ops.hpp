#pragma once
#include <vector>
#include <span>

namespace cpu {
    enum ReductionOp { MIN, MAX, SUM };

    std::vector<double> matmul(const std::vector<double> &A, const std::vector<double> &B, int K, int X, int Y);
    std::vector<double> binary_op(const char op, const std::vector<double>& matrix_A, const std::vector<double>& matrix_B, int size);
    std::vector<double> softmax(const std::vector<double>& matrix_A, int N, int C);
    double reduction(const ReductionOp op, const std::span<const double>& input);
}