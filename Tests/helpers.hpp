#pragma once
#include <gtest/gtest.h>
#include <vector>
#include "tensor.hpp"

inline Tensor make_tensor(const std::vector<std::vector<double>>& data) {
    Tensor t({static_cast<int>(data.size()),
              static_cast<int>(data[0].size())});
    for (int i = 0; i < data.size(); ++i)
        for (int j = 0; j < data[i].size(); ++j)
            t[i][j] = data[i][j];
    return t;
}

inline void expect_flatten_tensor_near(
    Tensor t,
    const std::vector<double>& expected,
    bool grad = false,
    double eps = 1e-4
) {
    ASSERT_EQ(t.total_count, expected.size());
    for (int i = 0; i < t.total_count; ++i) {
        double v = grad ? t.values[i]->grad : t.values[i]->data;
        EXPECT_NEAR(v, expected[i], eps);
    }
}
