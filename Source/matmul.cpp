#include "matmul.hpp"

std::vector<double> naive_matmul(const std::vector<double> &A, const std::vector<double> &B, int K, int X, int Y)
{
    std::vector<double> output(X * Y, 0.0);

    for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
            for (int k = 0; k < K; ++k)
                output[y * X + x] += A[y * K + k] * B[k * X + x];

    return output;
}

std::vector<double> _matmul(const std::vector<double> A, const std::vector<double> B, int K, int X, int Y, const std::string device)
{
    if (device == "cpu")
        return naive_matmul(A, B, K, X, Y);

    return {};
}
