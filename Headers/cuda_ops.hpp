#pragma once
#include <vector>
#include <span>
#include "globals.hpp"

namespace cuda {
    ::std::vector<float> matmul(const ::std::vector<float>& matrix_A, const ::std::vector<float>& matrix_B, int K, int X, int Y);
    ::std::vector<float> binary_op(const char op, const ::std::vector<float>& matrix_A, const ::std::vector<float>& matrix_B, int size);
    void softmax(const ::std::vector<float>& input, float* output, int N, int C);

    float reduction(const ReductionOp op, const ::std::span<const float>& input);
    float full_reduction(const ReductionOp op, const ::std::span<const float>& input);
}