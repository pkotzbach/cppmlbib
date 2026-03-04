#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tensor.hpp"
#include "cuda_debug.h"
#include <tuple>
#include <string>
#include <vector>

using ::testing::FloatNear;
using ::testing::Pointwise;

enum class Op { ADD, SUB, MUL, DIV };

// Parameterizing by both Device (string) and Operator (Op)
class BroadcastingTest : public ::testing::TestWithParam<std::tuple<std::string, Op>> {
protected:
    void SetUp() override {
#ifdef CUDA_TEST
        if (std::get<0>(GetParam()) == "cuda") g_cuda_kernel_launches = 0;
#endif
    }
    void TearDown() override {
#ifdef CUDA_TEST
        if (std::get<0>(GetParam()) == "cuda") EXPECT_GT(g_cuda_kernel_launches, 0);
#endif
    }

    Tensor_ptr apply_op(Tensor_ptr a, Tensor_ptr b) {
        Op op = std::get<1>(GetParam());
        switch (op) {
            case Op::ADD: return a + b;
            case Op::SUB: return a - b;
            case Op::MUL: return a * b;
            case Op::DIV: return a / b;
        }
        return nullptr;
    }
};

TEST_P(BroadcastingTest, Scalar)
{
    std::string device = std::get<0>(GetParam());
    Op op = std::get<1>(GetParam());

    std::vector<float> a_vals = {1, 2, 3, 4};
    std::vector<float> b_vals = {10.0};
    Tensor_ptr a = Tensor::init({2, 2}, a_vals, device);
    Tensor_ptr b = Tensor::init({1}, b_vals, device);   // scalar

    auto r = apply_op(a, b);

    // Dynamic expectation generators for values and gradients
    std::vector<float> exp_v(4), exp_ga(4), exp_gb(1, 0.0);
    for(int i = 0; i < 4; ++i) {
        if (op == Op::ADD) { exp_v[i] = a_vals[i] + b_vals[0]; exp_ga[i] = 1; exp_gb[0] += 1; }
        else if (op == Op::SUB) { exp_v[i] = a_vals[i] - b_vals[0]; exp_ga[i] = 1; exp_gb[0] -= 1; }
        else if (op == Op::MUL) { exp_v[i] = a_vals[i] * b_vals[0]; exp_ga[i] = b_vals[0]; exp_gb[0] += a_vals[i]; }
        else if (op == Op::DIV) { exp_v[i] = a_vals[i] / b_vals[0]; exp_ga[i] = 1.0/b_vals[0]; exp_gb[0] += -a_vals[i] / (b_vals[0] * b_vals[0]); }
    }

    EXPECT_THAT(r->values_vec(), Pointwise(FloatNear(1e-5), exp_v));

    r->sum()->backward();

    EXPECT_THAT(a->grads_vec(), Pointwise(FloatNear(1e-5), exp_ga));
    EXPECT_THAT(b->grads_vec(), Pointwise(FloatNear(1e-5), exp_gb));
}

TEST_P(BroadcastingTest, RowVector)
{
    std::string device = std::get<0>(GetParam());
    Op op = std::get<1>(GetParam());

    std::vector<float> a_vals = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_vals = {10, 20, 30};
    Tensor_ptr a = Tensor::init({2, 3}, a_vals, device);
    Tensor_ptr b = Tensor::init({3}, b_vals, device);

    auto r = apply_op(a, b);

    std::vector<float> exp_v(6), exp_ga(6), exp_gb(3, 0.0);
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            int idx = i * 3 + j;
            if (op == Op::ADD) { exp_v[idx] = a_vals[idx] + b_vals[j]; exp_ga[idx] = 1; exp_gb[j] += 1; }
            else if (op == Op::SUB) { exp_v[idx] = a_vals[idx] - b_vals[j]; exp_ga[idx] = 1; exp_gb[j] -= 1; }
            else if (op == Op::MUL) { exp_v[idx] = a_vals[idx] * b_vals[j]; exp_ga[idx] = b_vals[j]; exp_gb[j] += a_vals[idx]; }
            else if (op == Op::DIV) { exp_v[idx] = a_vals[idx] / b_vals[j]; exp_ga[idx] = 1.0/b_vals[j]; exp_gb[j] += -a_vals[idx] / (b_vals[j] * b_vals[j]); }
        }
    }

    EXPECT_THAT(r->values_vec(), Pointwise(FloatNear(1e-5), exp_v));

    r->sum()->backward();

    EXPECT_THAT(a->grads_vec(), Pointwise(FloatNear(1e-5), exp_ga));
    EXPECT_THAT(b->grads_vec(), Pointwise(FloatNear(1e-5), exp_gb));
}

TEST_P(BroadcastingTest, Column)
{
    std::string device = std::get<0>(GetParam());
    Op op = std::get<1>(GetParam());

    std::vector<float> a_vals = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_vals = {10, 20};
    Tensor_ptr a = Tensor::init({2, 3}, a_vals, device);
    Tensor_ptr b = Tensor::init({2, 1}, b_vals, device);

    auto r = apply_op(a, b);

    std::vector<float> exp_v(6), exp_ga(6), exp_gb(2, 0.0);
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            int idx = i * 3 + j;
            if (op == Op::ADD) { exp_v[idx] = a_vals[idx] + b_vals[i]; exp_ga[idx] = 1; exp_gb[i] += 1; }
            else if (op == Op::SUB) { exp_v[idx] = a_vals[idx] - b_vals[i]; exp_ga[idx] = 1; exp_gb[i] -= 1; }
            else if (op == Op::MUL) { exp_v[idx] = a_vals[idx] * b_vals[i]; exp_ga[idx] = b_vals[i]; exp_gb[i] += a_vals[idx]; }
            else if (op == Op::DIV) { exp_v[idx] = a_vals[idx] / b_vals[i]; exp_ga[idx] = 1.0/b_vals[i]; exp_gb[i] += -a_vals[idx] / (b_vals[i] * b_vals[i]); }
        }
    }

    EXPECT_THAT(r->values_vec(), Pointwise(FloatNear(1e-5), exp_v));

    r->sum()->backward();

    EXPECT_THAT(a->grads_vec(), Pointwise(FloatNear(1e-5), exp_ga));
    EXPECT_THAT(b->grads_vec(), Pointwise(FloatNear(1e-5), exp_gb));
}

TEST_P(BroadcastingTest, HighDimBroadcast)
{
    std::string device = std::get<0>(GetParam());
    Op op = std::get<1>(GetParam());

    std::vector<float> a_vals = {
        1,2,3,4,     5,6,7,8,     9,10,11,12,
        13,14,15,16, 17,18,19,20, 21,22,23,24
    };
    std::vector<float> b_vals = {1, 10, 100, 1000};
    Tensor_ptr a = Tensor::init({2, 3, 4}, a_vals, device);
    Tensor_ptr b = Tensor::init({4}, b_vals, device);

    auto r = apply_op(a, b);

    std::vector<float> exp_v(24), exp_ga(24), exp_gb(4, 0.0);
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            for(int k = 0; k < 4; ++k) {
                int idx = i * 12 + j * 4 + k;
                if (op == Op::ADD) { exp_v[idx] = a_vals[idx] + b_vals[k]; exp_ga[idx] = 1; exp_gb[k] += 1; }
                else if (op == Op::SUB) { exp_v[idx] = a_vals[idx] - b_vals[k]; exp_ga[idx] = 1; exp_gb[k] -= 1; }
                else if (op == Op::MUL) { exp_v[idx] = a_vals[idx] * b_vals[k]; exp_ga[idx] = b_vals[k]; exp_gb[k] += a_vals[idx]; }
                else if (op == Op::DIV) { exp_v[idx] = a_vals[idx] / b_vals[k]; exp_ga[idx] = 1.0/b_vals[k]; exp_gb[k] += -a_vals[idx] / (b_vals[k] * b_vals[k]); }
            }
        }
    }

    EXPECT_THAT(r->values_vec(), Pointwise(FloatNear(1e-5), exp_v));

    r->sum()->backward();

    EXPECT_THAT(a->grads_vec(), Pointwise(FloatNear(1e-5), exp_ga));
    EXPECT_THAT(b->grads_vec(), Pointwise(FloatNear(1e-5), exp_gb));
}

TEST_P(BroadcastingTest, BroadcastWithTranspose)
{
    std::string device = std::get<0>(GetParam());
    Op op = std::get<1>(GetParam());

    std::vector<float> a_vals = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_vals = {10, 20, 30};
    Tensor_ptr a = Tensor::init({2, 3}, a_vals, device);
    Tensor_ptr b = Tensor::init({3, 1}, b_vals, device);
    Tensor_ptr bt = b->transpose();  // shape (1, 3), stride swap

    auto r = apply_op(a, bt);

    std::vector<float> exp_v(6), exp_ga(6), exp_gb(3, 0.0);
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            int idx = i * 3 + j;
            if (op == Op::ADD) { exp_v[idx] = a_vals[idx] + b_vals[j]; exp_ga[idx] = 1; exp_gb[j] += 1; }
            else if (op == Op::SUB) { exp_v[idx] = a_vals[idx] - b_vals[j]; exp_ga[idx] = 1; exp_gb[j] -= 1; }
            else if (op == Op::MUL) { exp_v[idx] = a_vals[idx] * b_vals[j]; exp_ga[idx] = b_vals[j]; exp_gb[j] += a_vals[idx]; }
            else if (op == Op::DIV) { exp_v[idx] = a_vals[idx] / b_vals[j]; exp_ga[idx] = 1.0/b_vals[j]; exp_gb[j] += -a_vals[idx] / (b_vals[j] * b_vals[j]); }
        }
    }

    EXPECT_THAT(r->values_vec(), Pointwise(FloatNear(1e-5), exp_v));

    r->sum()->backward();

    EXPECT_THAT(a->grads_vec(), Pointwise(FloatNear(1e-5), exp_ga));
    EXPECT_THAT(bt->grads_vec(), Pointwise(FloatNear(1e-5), exp_gb));
}


// Optional formatter to assign easily readable suffix test names inside Google Test
std::string PrintTestParamName(const ::testing::TestParamInfo<std::tuple<std::string, Op>>& info) {
    std::string device = std::get<0>(info.param);
    Op op = std::get<1>(info.param);
    std::string op_str;
    switch (op) {
        case Op::ADD: op_str = "ADD"; break;
        case Op::SUB: op_str = "SUB"; break;
        case Op::MUL: op_str = "MUL"; break;
        case Op::DIV: op_str = "DIV"; break;
    }
    return device + "_" + op_str;
}

INSTANTIATE_TEST_SUITE_P(CPU, BroadcastingTest, 
    ::testing::Combine(
        ::testing::Values("cpu"),
        ::testing::Values(Op::ADD, Op::SUB, Op::MUL, Op::DIV)
    ),
    PrintTestParamName
);

#ifdef CUDA_TEST
INSTANTIATE_TEST_SUITE_P(CUDA, BroadcastingTest, 
    ::testing::Combine(
        ::testing::Values("cuda"),
        ::testing::Values(Op::ADD, Op::SUB, Op::MUL, Op::DIV)
    ),
    PrintTestParamName
);
#endif