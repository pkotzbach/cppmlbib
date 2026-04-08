#include "test_common.hpp"
#include "cuda_ops.hpp"
#include <cuda_runtime.h>

TEST(CudaOpsTest, Transpose)
{
    int N = 3;
    int C = 4;

    // 0 1 2  3
    // 4 5 6  7
    // 8 9 10 11
    float* a = new float[N * C];
    for (int i = 0; i < N * C; ++i) a[i] = i;

    // 0 4 8
    // 1 5 9
    // 2 6 10
    // 3 7 11
    std::vector<float> aT(C * N);
    cuda::transpose(a, aT.data(), N, C);

    EXPECT_THAT(aT,
            Pointwise(FloatNear(1e-5),
                      std::vector<float>{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}));
    
    delete[] a;
}

TEST(CudaOpsTest, Matmul_311x283_283x227)
{
    int Y = 311;
    int K = 283;
    int X = 227;

    std::vector<float> a(Y * K);
    std::vector<float> b(K * X);
    for (int i = 0; i < Y * K; ++i) a[i] = static_cast<float>(i % 100) / 100.0f;
    for (int i = 0; i < K * X; ++i) b[i] = static_cast<float>(i % 100) / 100.0f;

    std::vector<float> expected(Y * X, 0.0f);
    for (int y = 0; y < Y; ++y) {
        for (int k = 0; k < K; ++k) {
            float a_val = a[y * K + k];
            for (int x = 0; x < X; ++x) {
                expected[y * X + x] += a_val * b[k * X + x];
            }
        }
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, Y * K * sizeof(float));
    cudaMalloc(&d_b, K * X * sizeof(float));
    cudaMalloc(&d_c, Y * X * sizeof(float));

    cudaMemcpy(d_a, a.data(), Y * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), K * X * sizeof(float), cudaMemcpyHostToDevice);

    cuda::matmul(d_a, d_b, d_c, K, X, Y);

    std::vector<float> result(Y * X);
    cudaMemcpy(result.data(), d_c, Y * X * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_THAT(result, Pointwise(FloatNear(1e-3), expected));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
