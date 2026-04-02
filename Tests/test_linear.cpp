#include "test_common.hpp"
#include "linear.hpp"

class LinearTest : public BaseDeviceTest {};

TEST_P(LinearTest, Forward) {
    std::string device = GetParam();
    Linear l1(5, 5, device);

    for (int o = 0; o < l1.get_out_size(); ++o) {
        l1.get_biases()->set(o, 0.1);
        for (int i = 0; i < l1.get_in_size(); ++i)
            l1.get_weights()->set({i, o}, 0.1 * o + 0.01 * i);
    }

    Tensor_ptr input = Tensor::init({2, 5}, {0.1, 0.2, 0.3, 0.4, 0.5, 
                                             1.1, -1.2, 0.8, 0.2, 0.0}, device);
    Tensor_ptr output = l1.forward(input);

    EXPECT_THAT(output->values_vec(),
        Pointwise(FloatNear(1e-5),
                std::vector<float>({
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
            Pointwise(FloatNear(1e-4),
                      std::vector<float>({0.3420, 0.3780, 0.2800})));
}

TEST_P(LinearTest, ConvolutionForward) {
    std::string device = GetParam();
    int in_channels = 2;
    int out_channels = 2;
    int kernel_size = 2;
    int stride = 1;
    int padding = 0;

    Convolution conv(in_channels, out_channels, kernel_size, stride, padding, device);

    Tensor_ptr input = Tensor::init(
        {1, 2, 3, 3},
        {
            // channel 0
            1,2,3,
            4,5,6,
            7,8,9,
            // channel 1
            10,11,12,
            13,14,15,
            16,17,18
        },
        device
    );

    // output ch0
    conv.get_weights()->set({0, 0}, 1.0f);
    conv.get_weights()->set({1, 0}, 0.0f);
    conv.get_weights()->set({2, 0}, 0.0f);
    conv.get_weights()->set({3, 0}, 1.0f);
    conv.get_weights()->set({4, 0}, 1.0f);
    conv.get_weights()->set({5, 0}, 0.0f);
    conv.get_weights()->set({6, 0}, 0.0f);
    conv.get_weights()->set({7, 0}, 1.0f);

    // output ch1
    for (int i = 0; i < 8; i++) {
        conv.get_weights()->set({i, 1}, 1.0f);
    }

    // biases
    conv.get_biases()->set({0, 0}, 0.5f);
    conv.get_biases()->set({0, 1}, 1.0f);

    Tensor_ptr output = conv.forward(input);

    EXPECT_EQ(output->get_shape(), std::vector<int>({1, 2, 2, 2}));

    std::vector<float> expected = {
        30.5f, 61.0f,
        34.5f, 69.0f,
        42.5f, 85.0f,
        46.5f, 93.0f
    };

    EXPECT_THAT(output->values_vec(), Pointwise(FloatNear(1e-4), expected));
}

INSTANTIATE_TEST_SUITE_P(CPU, LinearTest, ::testing::Values("cpu"));
#ifdef CUDA_TEST
INSTANTIATE_TEST_SUITE_P(CUDA, LinearTest, ::testing::Values("cuda"));
#endif

