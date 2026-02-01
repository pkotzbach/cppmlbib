#include <gtest/gtest.h>
#include "value.hpp"

TEST(ValueTest, BasicOpsAndBackward) {
    auto x = std::make_shared<Value>(5.0);
    auto y = std::make_shared<Value>(6.0);
    auto z = std::make_shared<Value>(-0.1);

    auto a = x * y;
    auto b = a * z;
    auto c = (b + z)->relu();
    auto e = b + c + a;
    auto final = x / y + z->exp() + e;

    final->backward();

    EXPECT_NEAR(x->grad, 5.567, 0.001);
    EXPECT_NEAR(y->grad, 4.361, 0.001);
    EXPECT_NEAR(z->grad, 30.905, 0.001);
    EXPECT_NEAR(final->data, 28.738, 0.001);

    final->zero_grad();
    EXPECT_NEAR(x->grad, 0.0, 1e-6);
    EXPECT_NEAR(y->grad, 0.0, 1e-6);
    EXPECT_NEAR(z->grad, 0.0, 1e-6);
}
