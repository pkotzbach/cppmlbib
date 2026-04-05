#include "test_common.hpp"
#include <cuda_runtime.h>
#include "cpu_ops.hpp"

TEST(CpuOpsTest, BTMatmul)
{
    // A
    // 4 8     2
    // 6 12.1 -2
    Tensor_ptr a = Tensor::init({2, 3}, {4.0, 8.0, 2.0, 6.0, 12.1, -2}, Device::CPU);

    // B
    // 2 1 1
    // 2 3 0.4

    // B^T
    // 2 2
    // 1 3
    // 1 0.4
    Tensor_ptr b = Tensor::init({2, 3}, {2.0, 1.0, 1.0, 2.0, 3.0, 0.4}, Device::CPU);

    auto result = cpu::BT_matmul(a->raw_values(), b->raw_values(), 3, 2, 2);

    EXPECT_THAT(result,
            Pointwise(FloatNear(1e-5),
                      std::vector<float>{18.0, 32.8, 22.1, 47.5}));
}

TEST(CpuOpsTest, BTMatmul_256x256)
{
    const int M = 256;
    const int K = 256;
    const int N = 256;

    std::vector<float> a_vals(M * K);
    std::vector<float> b_vals(N * K);

    for (int i = 0; i < M * K; ++i)
        a_vals[i] = static_cast<float>(i % 13 - 6);

    for (int i = 0; i < N * K; ++i)
        b_vals[i] = static_cast<float>((i * 2) % 17 - 8);

    Tensor_ptr a = Tensor::init({M, K}, a_vals, Device::CPU);
    Tensor_ptr b = Tensor::init({N, K}, b_vals, Device::CPU);

    auto result = cpu::BT_matmul(a->raw_values(), b->raw_values(), K, M, N);

    std::vector<float> expected(M * N, 0.0f);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += a_vals[i * K + k] * b_vals[j * K + k];
            }
            expected[i * N + j] = sum;
        }
    }

    EXPECT_THAT(result,
        Pointwise(FloatNear(1e-4), expected));
}

TEST(CpuOpsTest, ATMatmul)
{
    Tensor_ptr a = Tensor::init({3, 2}, {2.0, 2.0, 1.0, 3.0, 1.0, 0.4}, Device::CPU);
    Tensor_ptr b = Tensor::init({3, 2}, {4.0, 6.0, 8.0, 12.1, 2.0, -2.0}, Device::CPU);
    auto result = cpu::AT_matmul(a->raw_values(), b->raw_values(), 3, 2, 2);

    EXPECT_THAT(result,
            Pointwise(FloatNear(1e-5),
                      std::vector<float>{18.0, 22.1, 32.8, 47.5}));
}

TEST(CpuOpsTest, ATMatmul_256x256)
{
    const int K = 256;
    const int M = 256;
    const int N = 256;

    std::vector<float> a_vals(K * M);
    std::vector<float> b_vals(K * N);

    for (int i = 0; i < K * M; ++i)
        a_vals[i] = static_cast<float>(i % 13 - 6);

    for (int i = 0; i < K * N; ++i)
        b_vals[i] = static_cast<float>((i * 2) % 17 - 8);

    auto result = cpu::AT_matmul(a_vals.data(), b_vals.data(), K, M, N);

    std::vector<float> expected(M * N, 0.0f);

    for (int k = 0; k < K; ++k)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                expected[m * N + n] += a_vals[k * M + m] * b_vals[k * N + n];
            }
        }
    }

    EXPECT_THAT(result,
        Pointwise(FloatNear(1e-4), expected));
}