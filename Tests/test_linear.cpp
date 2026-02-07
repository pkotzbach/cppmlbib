#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "linear.hpp"
#include "tensor.hpp"

using ::testing::DoubleNear;
using ::testing::Pointwise;

TEST(LinearTest, Forward) {
    Linear l1(5, 5);

    for (int o = 0; o < l1.out_size; ++o) {
        l1.biases->at(o) = 0.1;
        for (int i = 0; i < l1.in_size; ++i)
            l1.weights->at(o * l1.in_size + i) = 0.1 * o + 0.01 * i;
    }

    Tensor_ptr input = Tensor::init({2, 5}, {0.1, 0.2, 0.3, 0.4, 0.5, 1.1, -1.2, 0.8, 0.2, 0.0});
    Tensor_ptr output = l1.forward(input);

    EXPECT_THAT(output->values_vec(),
            Pointwise(DoubleNear(1e-6),
                      std::vector<double>({0.14, 0.29, 0.44, 0.59, 0.74})));
}

// TEST(LinearTest, CudaForward) {
//     Linear l1(5, 5, "cuda");

//     for (int o = 0; o < l1.out_size; ++o) {
//         l1.biases[o]->data = 0.1;
//         for (int i = 0; i < l1.in_size; ++i)
//             l1.weights[o][i]->data = 0.1 * o + 0.01 * i;
//     }

//     Tensor input = make_tensor({{0.1, 0.2, 0.3, 0.4, 0.5}});
//     Tensor output = l1.forward(input);

//     expect_flatten_tensor_near(
//         output, {0.14, 0.29, 0.44, 0.59, 0.74}
//     );
// }

TEST(SoftmaxTest, Forward) {
    Tensor_ptr input = Tensor::init({1, 3}, {0.1, 0.2, -0.1});
    Softmax s;

    Tensor_ptr output = s.forward(input);

    EXPECT_THAT(output->values_vec(),
            Pointwise(DoubleNear(1e-4),
                      std::vector<double>({0.3420, 0.3780, 0.2800})));
}
