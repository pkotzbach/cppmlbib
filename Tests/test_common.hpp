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

inline void cuda_reset(const std::string& device) {
#ifdef CUDA_TEST
    if (device == "cuda") g_cuda_kernel_launches = 0;
#endif
}

inline void cuda_check_launched(const std::string& device, bool expect_launch = true) {
#ifdef CUDA_TEST
    if (device == "cuda" && expect_launch) {
        EXPECT_GT(g_cuda_kernel_launches, 0);
    }
#endif
}

} // namespace test_utils

class BaseDeviceTest : public ::testing::TestWithParam<std::string> {
protected:
    bool expect_cuda_launch = true;

    void SetUp() override {
        test_utils::cuda_reset(GetParam());
        if (GetParam() == "cpu") expect_cuda_launch = false;
    }

    void TearDown() override {
        test_utils::cuda_check_launched(GetParam(), expect_cuda_launch);
    }
};
