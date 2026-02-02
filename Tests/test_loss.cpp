#include <gtest/gtest.h>
#include "loss.hpp"
#include "test_helpers.hpp"

TEST(LossTest, MSE) {
    Tensor input = make_tensor({{0.1, 0.2, -0.1}});
    Tensor target = make_tensor({{1, 0, -1}});

    Value_ptr loss = MSELoss(input, target);

    EXPECT_NEAR(loss->data, 0.5533, 1e-4);

    loss->backward();
    expect_flatten_tensor_near(
        input, {-0.6, 0.1333, 0.6}, true
    );
}
