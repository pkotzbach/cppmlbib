#include <gtest/gtest.h>
#include "tensor.hpp"
#include "helpers.hpp"

TEST(TensorTest, ConstructionAndIndexing) {
    Tensor t({2, 3, 4});

    EXPECT_EQ(t.shape, std::vector<int>({2, 3, 4}));
    EXPECT_EQ(t.total_count, 24);
    EXPECT_EQ(t.values.size(), 24);

    EXPECT_THROW(Tensor(std::vector<int>{}), std::exception);

    EXPECT_EQ(t[0].dim, 1);
    EXPECT_EQ(t[1][2].dim, 2);

    EXPECT_THROW(t[5], std::exception);

    t[0][0][0] = 1.5;
    t[1][1][3] = 3.7;

    EXPECT_NEAR(t.values[0]->data, 1.5, 1e-6);
    EXPECT_NEAR(t.values[19]->data, 3.7, 1e-6);
}

TEST(TensorTest, ElementwiseOps) {
    Tensor t1 = make_tensor({{0.1, 0.2, -0.1}, {1, 2, 3}});
    Tensor t2 = make_tensor({{0.3, 0.6, -0.23}, {-1, -2, -3}});

    expect_flatten_tensor_near(
        t1 + t2, {0.4, 0.8, -0.33, 0, 0, 0}
    );
    expect_flatten_tensor_near(
        t1 - t2, {-0.2, -0.4, 0.13, 2, 4, 6}
    );
    expect_flatten_tensor_near(
        t1 * t2, {0.03, 0.12, 0.023, -1, -4, -9}
    );
}

TEST(TensorTest, ArgmaxLastDim) {
    Tensor input = make_tensor({{0.1, 0.2, -0.1}, {1, 1, 5}});
    Tensor result = input.argmax(1);

    expect_flatten_tensor_near(result, {1, 2});
}
