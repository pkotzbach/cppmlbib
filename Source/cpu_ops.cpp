#include "cpu_ops.hpp"

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

} // namespace cpu