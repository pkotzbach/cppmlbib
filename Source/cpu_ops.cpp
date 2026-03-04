#include "cpu_ops.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace cpu {

std::vector<double> matmul(const std::vector<double> &A, const std::vector<double> &B, int K, int X, int Y)
{
    std::vector<double> output(X * Y, 0.0);

    for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
            for (int k = 0; k < K; ++k)
                output[y * X + x] += A[y * K + k] * B[k * X + x];

    return output;
}

std::vector<double> binary_op(const char op, const std::vector<double>& matrix_A, const std::vector<double>& matrix_B, int size) {
    std::vector<double> result(size);
    for (int i = 0; i < size; ++i) {
        switch (op) {
            case '+': result[i] = matrix_A[i] + matrix_B[i]; break;
            case '-': result[i] = matrix_A[i] - matrix_B[i]; break;
            case '*': result[i] = matrix_A[i] * matrix_B[i]; break;
            case '/': result[i] = matrix_A[i] / matrix_B[i]; break;
        }
    }
    return result;
}

std::vector<double> softmax(const std::vector<double>& matrix_A, int N, int C) {
    std::vector<double> result(N * C);
    for (int i = 0; i < N; ++i) {
        double max_val = matrix_A[i * C];
        for (int j = 1; j < C; ++j) {
            max_val = std::max(max_val, matrix_A[i * C + j]);
        }

        double sum = 0.0;
        for (int j = 0; j < C; ++j) {
            result[i * C + j] = std::exp(matrix_A[i * C + j] - max_val);
            sum += result[i * C + j];
        }

        for (int j = 0; j < C; ++j) {
            result[i * C + j] /= sum;
        }
    }
    return result;
}

double reduction(const ReductionOp op, const std::span<const double>& input) {
    if (input.empty()) return 0.0;
    
    switch (op) {
        case MIN: return *std::min_element(input.begin(), input.end());
        case MAX: return *std::max_element(input.begin(), input.end());
        case SUM: return std::accumulate(input.begin(), input.end(), 0.0);
    }
    return 0.0;
}

} // namespace cpu