#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tensor.hpp"
#include <cuda_runtime.h>
#include "cuda_debug.h"

using ::testing::DoubleNear;
using ::testing::Pointwise;

class TensorTest : public ::testing::TestWithParam<std::string> {
protected:
    bool expect_cuda = true;
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

TEST_P(TensorTest, ConstructionAndIndexing)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr t = Tensor::init({2, 3, 4}, false, device);

    EXPECT_EQ(t->get_shape(), std::vector<int>({2, 3, 4}));
    EXPECT_EQ(t->get_total_count(), 24);

    EXPECT_THROW(t->at(std::vector<int>{5}), std::exception);

    t->at({0, 0, 0}) = 1.5;
    t->at({1, 1, 3}) = 3.7;

    EXPECT_NEAR(t->at(0), 1.5, 1e-6);
    EXPECT_NEAR(t->at(19), 3.7, 1e-6);
}

TEST_P(TensorTest, ArgmaxLastDim)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr input = Tensor::init({2, 3}, {0.1, 0.2, -0.1, 1, 1, 5}, device);
    Tensor_ptr result = input->argmax(1);

    EXPECT_THAT(result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1.0, 2.0}));
}

TEST_P(TensorTest, AdditionOperator)
{
    std::string device = GetParam();
    Tensor_ptr a = Tensor::init({2, 2}, {1.0, 2.0, 3.0, 4.0}, device);
    Tensor_ptr b = Tensor::init({2, 2}, {5.0, 6.0, 7.0, 8.0}, device);

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

TEST_P(TensorTest, SubtractionOperator)
{
    std::string device = GetParam();
    Tensor_ptr a = Tensor::init({2, 2}, {5.0, 6.0, 7.0, 8.0}, device);
    Tensor_ptr b = Tensor::init({2, 2}, {1.0, 2.0, 3.0, 4.0}, device);

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

TEST_P(TensorTest, MultiplicationOperator)
{
    std::string device = GetParam();
    Tensor_ptr a = Tensor::init({2, 2}, {1.0, 2.0, 3.0, 4.0}, device);
    Tensor_ptr b = Tensor::init({2, 2}, {5.0, 6.0, 7.0, 8.0}, device);

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

TEST_P(TensorTest, AssignmentOperator)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr a = Tensor::init({2, 2}, {1.0, 2.0, 3.0, 4.0}, device);
    Tensor_ptr b = a;  // operator=

    EXPECT_EQ(a.get(), b.get());
    EXPECT_EQ(b->get_shape(), std::vector<int>({2, 2}));

    b->at({0, 0}) = 42.0;

    EXPECT_NEAR(a->at({0, 0}), 42.0, 1e-6);
}

TEST_P(TensorTest, DivisionOperator)
{
    std::string device = GetParam();
    Tensor_ptr a = Tensor::init({2, 2}, {4.0, 8.0, 2.0, 6.0}, device);
    Tensor_ptr b = Tensor::init({2, 2}, {2.0, 2.0, 1.0, 3.0}, device);

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

TEST_P(TensorTest, SumOperation)
{
    std::string device = GetParam();
    Tensor_ptr t = Tensor::init({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, device);

    auto sum_result = t->sum();
    EXPECT_EQ(sum_result->values_vec().size(), 1);
    EXPECT_NEAR(sum_result->values_vec()[0], 21.0, 1e-6);

    sum_result->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorTest, SumAlongAxis0)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr t = Tensor::init({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, device);

    auto sum = t->sum(0);
    EXPECT_EQ(sum->get_shape().size(), 1);

    EXPECT_THAT(sum->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{9.0, 12.0}));

    sum->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorTest, SumAlongAxis1)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr t = Tensor::init({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, device);

    auto sum = t->sum(1);
    EXPECT_EQ(sum->get_shape().size(), 1);

    EXPECT_THAT(sum->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{3.0, 7.0, 11.0}));

    sum->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorTest, MaxOperation)
{
    std::string device = GetParam();
    Tensor_ptr t = Tensor::init({2, 3}, {1.0, 2.1, 7.0, -4.0, 5.0, 6.0}, device);

    auto max_result = t->max();
    EXPECT_EQ(max_result->values_vec().size(), 1);
    EXPECT_NEAR(max_result->values_vec()[0], 7.0, 1e-6);

    // max_result->backward();

    // EXPECT_THAT(t->grads_vec(),
    //             Pointwise(DoubleNear(1e-6),
    //                       std::vector<double>{1, 1, 1, 1, 1, 1}));
}


TEST_P(TensorTest, ReLUOperation)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr t = Tensor::init({3}, {1.0, -2.0, 0.5}, device);

    auto relu_result = t->relu();

    EXPECT_THAT(relu_result->values_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1.0, 0.0, 0.5}));

    relu_result->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-6),
                          std::vector<double>{1.0, 0.0, 1.0}));
}

TEST_P(TensorTest, SoftmaxOperation)
{
    std::string device = GetParam();
    Tensor_ptr a = Tensor::init({1, 3}, {0.1, 0.1, -0.1}, device);
    Tensor_ptr b = Tensor::init({1, 3}, {0.1, 1.4, 0.82}, device);
    Tensor_ptr softmax_result = a->softmax();

    EXPECT_THAT(softmax_result->values_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{0.3548, 0.3548, 0.2905}));

    (softmax_result * b)->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{-0.2378, 0.2234, 0.0144}));
    EXPECT_THAT(b->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{0.3548, 0.3548, 0.2905}));
}

TEST_P(TensorTest, SoftmaxOperationLargeNumbers)
{
    expect_cuda = false;
    std::string device = GetParam();

    Tensor_ptr a = Tensor::init({1, 3}, {1000.0, 1000.0, 999.0}, device);
    Tensor_ptr b = Tensor::init({1, 3}, {0.1, 1.4, 0.82}, device);

    Tensor_ptr softmax_result = a->softmax();

    EXPECT_THAT(softmax_result->values_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{0.4223, 0.4223, 0.1554}));

    (softmax_result * b)->sum()->backward();

    EXPECT_THAT(a->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{-0.2791,  0.2699,  0.0092}));
    
    EXPECT_THAT(b->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{0.4223, 0.4223, 0.1554}));

}

TEST_P(TensorTest, ExpOperation)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr t = Tensor::init({2}, {0.0, 1.0}, device);

    auto exp_result = t->exp();

    EXPECT_THAT(exp_result->values_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 2.71828}));

    exp_result->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 2.71828}));
}

TEST_P(TensorTest, ExpSmallValues)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr t = Tensor::init({3}, {0.0, 0.5, -0.5}, device);

    auto exp_result = t->exp();

    EXPECT_THAT(exp_result->values_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 1.6487, 0.6065}));

    exp_result->sum()->backward();

    EXPECT_THAT(t->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{1.0, 1.6487, 0.6065}));
}

TEST_P(TensorTest, ComplexExpression)
{
    std::string device = GetParam();
    Tensor_ptr x = Tensor::init({3}, {1.0, -2.0, 3.0}, device);
    Tensor_ptr y = Tensor::init({3}, {5.0, 1.4, 0.82}, device);

    auto result = ((x * y) + (x / y)->exp())->sum();
    result->sum()->backward();

    EXPECT_THAT(x->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{5.2443, 1.5712, 48.1426}));

    EXPECT_THAT(y->grads_vec(),
                Pointwise(DoubleNear(1e-4),
                          std::vector<double>{0.9511, -1.7555, -170.1314}));
}

TEST_P(TensorTest, Transpose)
{
    expect_cuda = false;
    std::string device = GetParam();
    Tensor_ptr x = Tensor::init({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, device);
    Tensor_ptr xT = x->transpose();
    
    EXPECT_EQ(xT->get_shape(), std::vector<int>({2, 3}));
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

TEST_P(TensorTest, Matmul)
{
    std::string device = GetParam();

    Tensor_ptr a = Tensor::init({2, 3}, {4.0, 8.0, 2.0, 6.0, 12.1, -2}, device);
    Tensor_ptr b = Tensor::init({3, 2}, {2.0, 2.0, 1.0, 3.0, 1.0, 0.4}, device);

    auto result = a->matmul(b);
    
    EXPECT_EQ(result->get_device(), device);
    EXPECT_EQ(result->get_shape(), std::vector<int>({2, 2}));
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


INSTANTIATE_TEST_SUITE_P(CPU, TensorTest, ::testing::Values("cpu"));
#ifdef CUDA_TEST
INSTANTIATE_TEST_SUITE_P(CUDA, TensorTest, ::testing::Values("cuda"));
#endif

