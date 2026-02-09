#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tensor.hpp"

using ::testing::DoubleNear;
using ::testing::Pointwise;

TEST(TensorTest, ConstructionAndIndexing)
{
    Tensor_ptr t = Tensor::init({2, 3, 4});

    EXPECT_EQ(t->shape, std::vector<int>({2, 3, 4}));
    EXPECT_EQ(t->total_count, 24);

    EXPECT_THROW(t->at(std::vector<int>{5}), std::exception);

    t->at({0, 0, 0}) = 1.5;
    t->at({1, 1, 3}) = 3.7;

    EXPECT_NEAR(t->at(0), 1.5, 1e-6);
    EXPECT_NEAR(t->at(19), 3.7, 1e-6);
}

TEST(TensorTest, ArgmaxLastDim)
{
    Tensor_ptr input = Tensor::init({2, 3}, {0.1, 0.2, -0.1, 1, 1, 5});
    Tensor_ptr result = input->argmax(1);

    EXPECT_THAT(result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1.0, 2.0}));
}

TEST(TensorTest, AdditionOperator)
{
    Tensor_ptr a = Tensor::init({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor_ptr b = Tensor::init({2, 2}, {5.0, 6.0, 7.0, 8.0});

    auto result = a + b;

    EXPECT_THAT(result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{6.0, 8.0, 10.0, 12.0}));

    result->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1}));

    EXPECT_THAT(b->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1}));
}

TEST(TensorTest, SubtractionOperator)
{
    Tensor_ptr a = Tensor::init({2, 2}, {5.0, 6.0, 7.0, 8.0});
    Tensor_ptr b = Tensor::init({2, 2}, {1.0, 2.0, 3.0, 4.0});

    auto result = a - b;

    EXPECT_THAT(result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{4.0, 4.0, 4.0, 4.0}));

    result->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1}));

    EXPECT_THAT(b->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{-1, -1, -1, -1}));
}

TEST(TensorTest, MultiplicationOperator)
{
    Tensor_ptr a = Tensor::init({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor_ptr b = Tensor::init({2, 2}, {5.0, 6.0, 7.0, 8.0});

    auto result = a * b;

    EXPECT_THAT(result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{5.0, 12.0, 21.0, 32.0}));

    result->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{5.0, 6.0, 7.0, 8.0}));

    EXPECT_THAT(b->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1.0, 2.0, 3.0, 4.0}));
}

TEST(TensorTest, AssignmentOperator)
{
    Tensor_ptr a = Tensor::init({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor_ptr b = a;  // operator=

    EXPECT_EQ(a.get(), b.get());
    EXPECT_EQ(b->shape, std::vector<int>({2, 2}));

    b->at({0, 0}) = 42.0;

    EXPECT_NEAR(a->at({0, 0}), 42.0, 1e-6);
}

TEST(TensorTest, DivisionOperator)
{
    Tensor_ptr a = Tensor::init({2, 2}, {4.0, 8.0, 2.0, 6.0});
    Tensor_ptr b = Tensor::init({2, 2}, {2.0, 2.0, 1.0, 3.0});

    auto result = a / b;

    EXPECT_THAT(result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{2.0, 4.0, 2.0, 2.0}));

    result->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
                Pointwise(DoubleNear(1e-5),
                          std::vector<double>{0.5, 0.5, 1.0, 0.333333}));

    EXPECT_THAT(b->grads_vec(),
                Pointwise(DoubleNear(1e-5),
                          std::vector<double>{-1.0, -2.0, -2.0, -0.666667}));
}

TEST(TensorTest, SumOperation)
{
    Tensor_ptr t = Tensor::init({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    auto sum_result = t->sum();
    EXPECT_EQ(sum_result->values_vec().size(), 1);
    EXPECT_NEAR(sum_result->values_vec()[0], 21.0, 1e-6);

    sum_result->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1, 1, 1}));
}

TEST(TensorTest, SumAlongAxis0)
{
    Tensor_ptr t = Tensor::init({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    auto sum = t->sum(0);
    EXPECT_EQ(sum->shape.size(), 1);

    EXPECT_THAT(sum->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{9.0, 12.0}));

    sum->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1, 1, 1}));
}

TEST(TensorTest, SumAlongAxis1)
{
    Tensor_ptr t = Tensor::init({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    auto sum = t->sum(1);
    EXPECT_EQ(sum->shape.size(), 1);

    EXPECT_THAT(sum->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{3.0, 7.0, 11.0}));

    sum->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1, 1, 1}));
}

TEST(TensorTest, ReLUOperation)
{
    Tensor_ptr t = Tensor::init({3}, {1.0, -2.0, 0.5});

    auto relu_result = t->relu();

    EXPECT_THAT(relu_result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1.0, 0.0, 0.5}));

    relu_result->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1.0, 0.0, 1.0}));
}

TEST(TensorTest, ExpOperation)
{
    Tensor_ptr t = Tensor::init({2}, {0.0, 1.0});

    auto exp_result = t->exp();

    EXPECT_THAT(exp_result->values_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 2.71828}));

    exp_result->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 2.71828}));
}

TEST(TensorTest, ExpSmallValues)
{
    Tensor_ptr t = Tensor::init({3}, {0.0, 0.5, -0.5});

    auto exp_result = t->exp();

    EXPECT_THAT(exp_result->values_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 1.6487, 0.6065}));

    exp_result->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 1.6487, 0.6065}));
}

TEST(TensorTest, ComplexExpression)
{
    Tensor_ptr x = Tensor::init({3}, {1.0, -2.0, 3.0});
    Tensor_ptr y = Tensor::init({3}, {5.0, 1.4, 0.82});

    auto result = ((x * y) + (x / y)->exp())->sum();
    result->sum()->backward();

    EXPECT_THAT(x->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{5.2443, 1.5712, 48.1426}));

    EXPECT_THAT(y->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{0.9511, -1.7555, -170.1314}));
}

TEST(TensorTest, Transpose)
{
    Tensor_ptr x = Tensor::init({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor_ptr xT = x->transpose();
    
    EXPECT_EQ(xT->shape, std::vector<int>({2, 3}));
    EXPECT_THAT(xT->values_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 3.0, 5.0, 2.0, 4.0, 6.0}));
    
    xT->zero_grad();
    xT->grad_at({0, 0}) = 1;
    xT->grad_at({0, 1}) = 2;
    xT->grad_at({0, 2}) = 3;
    xT->grad_at({1, 0}) = 4;
    xT->grad_at({1, 1}) = 5;
    xT->grad_at({1, 2}) = 6;
    xT->sum()->backward();
    
    // TODO: not sure if this test is correct
    EXPECT_THAT(x->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{2, 5, 3, 6, 4, 7}));
}

TEST(TensorTest, MatmulOperator)
{
    Tensor_ptr a = Tensor::init({2, 3}, {4.0, 8.0, 2.0, 6.0, 12.1, -2});
    Tensor_ptr b = Tensor::init({3, 2}, {2.0, 2.0, 1.0, 3.0, 1.0, 0.4});

    auto result = a->matmul(b);

    EXPECT_EQ(result->shape, std::vector<int>({2, 2}));
    EXPECT_THAT(result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{18.0, 32.8, 22.1, 47.5}));

    result->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
                Pointwise(DoubleNear(1e-5),
                          std::vector<double>{4.0, 4.0, 1.4, 4.0, 4.0, 1.4}));

    EXPECT_THAT(b->grads_vec(),
                Pointwise(DoubleNear(1e-5),
                          std::vector<double>{10.0, 10.0, 20.1, 20.1, 0.0, 0.0}));
}