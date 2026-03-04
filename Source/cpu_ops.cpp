#include "cpu_ops.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace cpu {

std::vector<float> matmul(const std::vector<float> &A, const std::vector<float> &B, int K, int X, int Y)
{
    std::vector<float> output(X * Y, 0.0);

    for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
            for (int k = 0; k < K; ++k)
                output[y * X + x] += A[y * K + k] * B[k * X + x];

    return output;
}
void softmax(const std::vector<float>& input, float* output, int N, int C) {
    for (int i = 0; i < N; ++i) {
        float max_val = input[i * C];
        for (int j = 1; j < C; ++j) {
            max_val = std::max(max_val, input[i * C + j]);
        }

        float sum = 0.0;
        for (int j = 0; j < C; ++j) {
            output[i * C + j] = std::exp(input[i * C + j] - max_val);
            sum += output[i * C + j];
        }

        for (int j = 0; j < C; ++j) {
            output[i * C + j] /= sum;
        }
    }
}

} // namespace cpu