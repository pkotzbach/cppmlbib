#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "optim.hpp"

using ::testing::DoubleNear;
using ::testing::Pointwise;

TEST(SGDTest, Step) {
    Tensor_ptr input = Tensor::init({3}, {0.1, 0.2, -0.1});
    input->grads[0] = 0.1;
    input->grads[1] = -0.1;
    input->grads[2] = 0.5;

    SGD optim({input}, 0.01);
    optim.step();

    EXPECT_THAT(input->values,
            Pointwise(DoubleNear(1e-6),
                      std::vector<double>({0.099, 0.201, -0.105})));
}
