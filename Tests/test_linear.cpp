#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "linear.hpp"
#include "tensor.hpp"
#include "cuda_debug.h"

using ::testing::DoubleNear;
using ::testing::Pointwise;

class LinearTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
#ifdef CUDA_TEST
        if (GetParam() == "cuda") g_cuda_kernel_launches = 0;
#endif
    }
    void TearDown() override {
#ifdef CUDA_TEST
        if (GetParam() == "cuda") EXPECT_GT(g_cuda_kernel_launches, 0);
#endif
    }
};

TEST_P(LinearTest, Forward) {
    std::string device = GetParam();
    Linear l1(5, 5, device);

    for (int o = 0; o < l1.get_out_size(); ++o) {
        l1.get_biases()->at(o) = 0.1;
        for (int i = 0; i < l1.get_in_size(); ++i)
            l1.get_weights()->at({i, o}) = 0.1 * o + 0.01 * i;
    }

    Tensor_ptr input = Tensor::init({2, 5}, {0.1, 0.2, 0.3, 0.4, 0.5, 
                                             1.1, -1.2, 0.8, 0.2, 0.0}, device);
    Tensor_ptr output = l1.forward(input);

    EXPECT_THAT(output->values_vec(),
        Pointwise(DoubleNear(1e-6),
                std::vector<double>({
                    0.14, 0.29, 0.44, 0.59, 0.74,
                    0.11, 0.20, 0.29, 0.38, 0.47
                })));
}

TEST_P(LinearTest, SoftmaxForward) {
    std::string device = GetParam();
    Tensor_ptr input = Tensor::init({1, 3}, {0.1, 0.2, -0.1}, device);
    Softmax s;

    Tensor_ptr output = s.forward(input);

    EXPECT_THAT(output->values_vec(),
            Pointwise(DoubleNear(1e-4),
                      std::vector<double>({0.3420, 0.3780, 0.2800})));
}

INSTANTIATE_TEST_SUITE_P(CPU, LinearTest, ::testing::Values("cpu"));
#ifdef CUDA_TEST
INSTANTIATE_TEST_SUITE_P(CUDA, LinearTest, ::testing::Values("cuda"));
#endif

