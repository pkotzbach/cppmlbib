#include <gtest/gtest.h>
#include "loss.hpp"
#include "cuda_debug.h"

class LossTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
#ifdef CUDA_TEST
        if (GetParam() == "cuda") g_cuda_kernel_launches = 0;
#endif
    }
    void TearDown() override {
#ifdef CUDA_TEST
        if (GetParam() == "cuda") EXPECT_GT(g_cuda_kernel_launches, 0);
#endif
    }
};

TEST_P(LossTest, MSE) {
    std::string device = GetParam();
    Tensor_ptr input = Tensor::init({2, 3}, {0.1, 0.2, -0.1, 0.3, 1, 0.2}, device);
    Tensor_ptr target = Tensor::init({2, 3}, {1, 0, -1, 1, 1, 1}, device);

    Tensor_ptr loss = MSELoss(input, target);

    EXPECT_NEAR(loss->values_vec()[0], 1.3950, 1e-4);

    loss->backward();
    EXPECT_EQ(input->grads_vec(), std::vector<float>({-0.9, 0.2, 0.9, -0.7, 0, -0.8}));
    EXPECT_EQ(target->grads_vec(), std::vector<float>({0.9, -0.2, -0.9, 0.7, 0, 0.8})); // TODO: target shouldnt have grad 
}

INSTANTIATE_TEST_SUITE_P(CPU, LossTest, ::testing::Values("cpu"));
#ifdef CUDA_TEST
INSTANTIATE_TEST_SUITE_P(CUDA, LossTest, ::testing::Values("cuda"));
#endif

