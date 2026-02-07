#include <gtest/gtest.h>
#include "loss.hpp"

TEST(LossTest, MSE) {
    Tensor_ptr input = Tensor::init({2, 3}, {0.1, 0.2, -0.1, 0.3, 1, 0.2});
    Tensor_ptr target = Tensor::init({2, 3}, {1, 0, -1, 1, 1, 1});

    Tensor_ptr loss = MSELoss(input, target);

    EXPECT_NEAR(loss->at(0), 1.3950, 1e-4);

    loss->backward();
    EXPECT_EQ(input->grads_vec(), std::vector<double>({-0.9, 0.2, 0.9, -0.7, 0, -0.8}));
    EXPECT_EQ(target->grads_vec(), std::vector<double>({0.9, -0.2, -0.9, 0.7, 0, 0.8})); // TODO: target shouldnt have grad 
}
