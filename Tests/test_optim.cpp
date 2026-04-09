#include "test_common.hpp"
#include "optim.hpp"

class SGDTest : public BaseDeviceTest {};

TEST_P(SGDTest, Step) {
    Device device = GetParam();
    Tensor_ptr input = Tensor::init({3}, {0.1, 0.2, -0.1}, device);
    input->grad_set(0, 0.1);
    input->grad_set(1, -0.1);
    input->grad_set(2, 0.5);

    SGD optim({input}, 0.01, device);
    optim.step();

    EXPECT_THAT(input->values_vec(),
            Pointwise(FloatNear(1e-5),
                      std::vector<float>({0.099, 0.201, -0.105})));
}

INSTANTIATE_TEST_SUITE_P(CPU, SGDTest, ::testing::Values(Device::CPU));
#ifdef CUDA_TEST
INSTANTIATE_TEST_SUITE_P(CUDA, SGDTest, ::testing::Values(Device::CUDA));
#endif

