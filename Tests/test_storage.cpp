#include "test_common.hpp"
#include <cuda_runtime.h>
#include "cuda_ops.hpp"

// TODO: make test that it breaks strides
TEST(StorageCudaTest, MakeContinous) {
    // Original:
    // 1 2
    // 3 4
    // 5 6
    // Transposed:
    // 1 3 5
    // 2 4 6
    
    std::vector<int> shape = {3, 2};
    std::vector<float> values = {1, 2, 3, 4, 5, 6};
    Tensor_ptr x = Tensor::init(shape, values, "cuda");
    auto xT = x->transpose();
    EXPECT_THAT(xT->values_vec(),
                Pointwise(FloatNear(1e-4),
                          std::vector<float>{1, 3, 5, 2, 4, 6}));
    EXPECT_THAT(x->values_vec(),
                Pointwise(FloatNear(1e-4),
                          values));
    // 1 2 3
    // 4 5 6

    // 2 5
    // 3 6
    // 4 7
    xT->zero_grad();
    xT->grad_set({0, 0}, 1);
    xT->grad_set({0, 1}, 2);
    xT->grad_set({0, 2}, 3);
    xT->grad_set({1, 0}, 4);
    xT->grad_set({1, 1}, 5);
    xT->grad_set({1, 2}, 6);
    xT->sum()->backward();
    cuda::make_continous(x);
    
    EXPECT_THAT(x->grads_vec(),
                Pointwise(FloatNear(1e-4),
                          std::vector<float>{2, 5, 3, 6, 4, 7}));
    EXPECT_THAT(xT->grads_vec(),
                Pointwise(FloatNear(1e-4),
                          std::vector<float>{2, 3, 4, 5, 6, 7}));
}
