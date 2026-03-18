#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "tensor.hpp"
#include "cuda_ops.hpp"
#include <vector>

TEST(StorageCudaTest, MakeContinous) {
    std::vector<int> shape = {2, 3};
    std::vector<float> values = {1, 2, 3, 4, 5, 6};
    
    // Original:
    // 1 2 3
    // 4 5 6
    // Transposed:
    // 1 4
    // 2 5
    // 3 6
    
    std::vector<int> transposed_shape = {3, 2};
    std::vector<int> transposed_strides = {1, 3};
    
    Storage storage("cuda", values, 6);
    cuda::make_continous(storage, transposed_strides, transposed_shape);
    auto result = storage.cpu();
    
    std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(result[i], expected[i]);
    }
}
