#pragma once
#include <vector>
#include <span>
#include "globals.hpp"

namespace cuda {
    ::std::vector<double> matmul(const ::std::vector<double>& matrix_A, const ::std::vector<double>& matrix_B, int K, int X, int Y);
    ::std::vector<double> binary_op(const char op, const ::std::vector<double>& matrix_A, const ::std::vector<double>& matrix_B, int size);
    void softmax(const ::std::vector<double>& input, double* output, int N, int C);

    double reduction(const ReductionOp op, const ::std::span<const double>& input);
    double full_reduction(const ReductionOp op, const ::std::span<const double>& input);
}