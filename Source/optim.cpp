#include "optim.hpp"
#include <stdio.h>
#include "cuda_kernels.h"

void Optimizer::zero_grad() {
    for (auto tensor : parameters) {
        for (int i = 0; i < tensor->get_total_count(); ++i) {
            tensor->grad_set(i, 0);
        }
    }
}
Optimizer::Optimizer(std::vector<std::vector<Tensor_ptr>> params, float lr, Device device) : lr(lr), device(device) {
    // flatten
    size_t total_size = 0;
    for (auto& vec : params) {
        total_size += vec.size();
    }
    parameters.reserve(total_size);

    for (auto& vec : params) {
        parameters.insert(parameters.end(), vec.begin(), vec.end());
    }
}

void SGD::step() {
    for (Tensor_ptr tensor : parameters) {
        float* values = tensor->raw_values();
        const float* grads = tensor->raw_grads();
        if (device == Device::CPU) {
            for (int i = 0; i < tensor->get_total_count(); ++i) {
                values[i] -= lr * grads[i];
            }
        }
        else if (device == Device::CUDA) {
            // cuda::sgd_step(values, grads, lr, tensor->get_total_count());
            cuda::sgd_step(values, grads, lr, tensor->get_total_count());
        }
    }
}
