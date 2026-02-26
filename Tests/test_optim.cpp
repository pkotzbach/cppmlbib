#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "optim.hpp"
#include "cuda_debug.h"

using ::testing::DoubleNear;
using ::testing::Pointwise;

class SGDTest : public ::testing::TestWithParam<std::string> {
protected:
    bool expect_cuda = false;
    void SetUp() override {
#ifdef CUDA_TEST
        if (GetParam() == "cuda") g_cuda_kernel_launches = 0;
#endif
    }
    void TearDown() override {
#ifdef CUDA_TEST
        if (GetParam() == "cuda" && expect_cuda) {
            EXPECT_GT(g_cuda_kernel_launches, 0);
        }
#endif
    }
};

TEST_P(SGDTest, Step) {
    expect_cuda = true;
    std::string device = GetParam();
    Tensor_ptr input = Tensor::init({3}, {0.1, 0.2, -0.1}, device);
    input->grad_at(0) = 0.1;
    input->grad_at(1) = -0.1;
    input->grad_at(2) = 0.5;

    SGD optim({input}, 0.01);
    optim.step();

    EXPECT_THAT(input->values_vec(),
            Pointwise(DoubleNear(1e-6),
                      std::vector<double>({0.099, 0.201, -0.105})));
}

INSTANTIATE_TEST_SUITE_P(CPU, SGDTest, ::testing::Values("cpu"));
#ifdef CUDA_TEST
INSTANTIATE_TEST_SUITE_P(CUDA, SGDTest, ::testing::Values("cuda"));
#endif

