#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tensor.hpp"

using ::testing::DoubleNear;
using ::testing::Pointwise;

TEST(BroadcastingTest, AddScalar)
{
    Tensor_ptr a = Tensor::init({2, 2}, {1, 2, 3, 4});
    Tensor_ptr b = Tensor::init(std::vector<int>({1}), std::vector<double>({10.0}));   // scalar

    auto r = a + b;

    EXPECT_THAT(r->values_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{11, 12, 13, 14}));

    r->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{1, 1, 1, 1}));

    EXPECT_THAT(b->grads_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{4}));
}

TEST(BroadcastingTest, AddRowVector)
{
    Tensor_ptr a = Tensor::init({2, 3},
        {1, 2, 3,
         4, 5, 6});

    Tensor_ptr b = Tensor::init({3}, {10, 20, 30});

    auto r = a + b;

    EXPECT_THAT(r->values_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{
                      11, 22, 33,
                      14, 25, 36}));

    r->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{1,1,1,1,1,1}));

    EXPECT_THAT(b->grads_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{2,2,2}));
}

TEST(BroadcastingTest, AddColumn)
{
    Tensor_ptr a = Tensor::init({2, 3},
        {1, 2, 3,
         4, 5, 6});

    Tensor_ptr b = Tensor::init({2, 1}, {10, 20});

    auto r = a + b;

    EXPECT_THAT(r->values_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{
                      11, 12, 13,
                      24, 25, 26}));

    r->sum()->backward();

    EXPECT_THAT(b->grads_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{3, 3}));
}

TEST(BroadcastingTest, AddHighDimBroadcast)
{
    Tensor_ptr a = Tensor::init({2, 3, 4},
        {
            1,2,3,4,     5,6,7,8,     9,10,11,12,
            13,14,15,16, 17,18,19,20, 21,22,23,24
        });

    Tensor_ptr b = Tensor::init({4}, {1, 10, 100, 1000});

    auto r = a + b;

    EXPECT_THAT(r->values_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{
                      2,12,103,1004,
                      6,16,107,1008,
                      10,20,111,1012,
                      14,24,115,1016,
                      18,28,119,1020,
                      22,32,123,1024
                  }));

    r->sum()->backward();

    EXPECT_THAT(b->grads_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{6,6,6,6}));
}

TEST(BroadcastingTest, AddBroadcastWithTranspose)
{
    Tensor_ptr a = Tensor::init({2, 3},
        {1, 2, 3,
         4, 5, 6});

    Tensor_ptr b = Tensor::init({3, 1}, {10, 20, 30});
    Tensor_ptr bt = b->transpose();  // shape (1, 3), stride swap

    auto r = a + bt;

    EXPECT_THAT(r->values_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{
                      11, 22, 33,
                      14, 25, 36}));

    r->sum()->backward();

    EXPECT_THAT(bt->grads_vec(),
        Pointwise(DoubleNear(1e-6),
                  std::vector<double>{2, 2, 2}));
}
