#include <gtest/gtest.h>
#include "linear.hpp"
#include "tensor.hpp"
#include "helpers.hpp"

TEST(LinearTest, Forward) {
    Linear l1(5, 5);

    for (int o = 0; o < l1.out_size; ++o) {
        l1.biases[o]->data = 0.1;
        for (int i = 0; i < l1.in_size; ++i)
            l1.weights[o][i]->data = 0.1 * o + 0.01 * i;
    }

    Tensor input = make_tensor({{0.1, 0.2, 0.3, 0.4, 0.5}});
    Tensor output = l1.forward(input);

    expect_flatten_tensor_near(
        output, {0.14, 0.29, 0.44, 0.59, 0.74}
    );
}

TEST(SoftmaxTest, ForwardAndBackward) {
    Tensor input = make_tensor({{0.1, 0.2, -0.1}});
    Softmax s;

    Tensor output = s.forward(input);
    expect_flatten_tensor_near(output, {0.3420, 0.3780, 0.2800});

    output[0][0]->backward();
    expect_flatten_tensor_near(
        input, {0.2250, -0.1293, -0.0958}, true
    );
}
