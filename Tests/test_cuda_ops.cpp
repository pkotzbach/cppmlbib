#include "test_common.hpp"
#include "cuda_ops.hpp"

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
}