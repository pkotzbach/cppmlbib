#include <gtest/gtest.h>
#include "optim.hpp"
#include "test_helpers.hpp"

TEST(SGDTest, Step) {
    Tensor input = make_tensor({{0.1, 0.2, -0.1}});
    input.values[0]->grad = 0.1;
    input.values[1]->grad = -0.1;
    input.values[2]->grad = 0.5;

    SGD optim({&input}, 0.01);
    optim.step();

    expect_flatten_tensor_near(
        input, {0.099, 0.201, -0.105}
    );
}
