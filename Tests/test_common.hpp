#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include "tensor.hpp"
#include "cuda_debug.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace test_utils {

inline void cuda_reset(Device device) {
#ifdef CUDA_TEST
    if (device == Device::CUDA) g_cuda_kernel_launches = 0;
#endif
}

inline void cuda_check_launched(Device device, bool expect_launch = true) {
#ifdef CUDA_TEST
    if (device == Device::CUDA && expect_launch) {
        EXPECT_GT(g_cuda_kernel_launches, 0);
    }
#endif
}

} // namespace test_utils

class BaseDeviceTest : public ::testing::TestWithParam<Device> {
protected:
    bool expect_cuda_launch = true;

    void SetUp() override {
        test_utils::cuda_reset(GetParam());
        if (GetParam() == Device::CPU) expect_cuda_launch = false;
    }

    void TearDown() override {
        test_utils::cuda_check_launched(GetParam(), expect_cuda_launch);
    }
};
