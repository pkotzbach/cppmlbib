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
    
    Tensor_ptr tensor = Tensor::init(shape, values, "cuda");
    tensor->set_shape(transposed_shape);
    tensor->set_strides(transposed_strides);
    cuda::make_continous(tensor);
    auto result = tensor->values_vec();
    
    std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(result[i], expected[i]);
    }
}
