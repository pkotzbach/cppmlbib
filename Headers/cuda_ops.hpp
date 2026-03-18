#pragma once
#include <vector>
#include <span>
#include "globals.hpp"
#include "tensor.hpp"

namespace cuda {
    ::std::vector<float> matmul(const ::std::vector<float>& matrix_A, const ::std::vector<float>& matrix_B, int K, int X, int Y);
    ::std::vector<float> matmul_wmma(const ::std::vector<float>& matrix_A, const ::std::vector<float>& matrix_B, int K, int X, int Y);
    ::std::vector<float> matmul_naive(const ::std::vector<float>& matrix_A, const ::std::vector<float>& matrix_B, int K, int X, int Y);
    ::std::vector<float> matmul_cublas(const ::std::vector<float>& matrix_A, const ::std::vector<float>& matrix_B, int K, int X, int Y);
    ::std::vector<float> binary_op(const char op, const ::std::vector<float>& matrix_A, const ::std::vector<float>& matrix_B, int size);
    void softmax(const float* input, float* output, int N, int C);

    float reduction(const ReductionOp op, const ::std::span<const float>& input);
    float full_reduction(const ReductionOp op, const ::std::span<const float>& input);

    void make_continous(Tensor_ptr tensor);
}