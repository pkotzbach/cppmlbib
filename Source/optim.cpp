#include "optim.hpp"
#include <stdio.h>

void Optimizer::zero_grad() {
    for (auto tensor : parameters) {
        for (int i = 0; i < tensor->get_total_count(); ++i) {
            tensor->grad_set(i, 0);
        }
    }
}
Optimizer::Optimizer(std::vector<std::vector<Tensor_ptr>> params, float lr) : lr(lr) {
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
        for (int i = 0; i < tensor->get_total_count(); ++i) {
            tensor->set(i, tensor->get(i) - lr * tensor->grad_get(i));
        }
    }
}
