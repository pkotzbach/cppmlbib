#include "optim.hpp"
#include <stdio.h>

void Optimizer::zero_grad()
{
    for (Tensor_ptr tensor : parameters)
    {
        for (int i = 0; i < tensor->total_count; ++i)
            tensor->grad_at(i) = 0;
    }
}

Optimizer::Optimizer(std::vector<std::vector<Tensor_ptr>> params, double lr) : lr(lr)
{
    // flatten
    size_t total_size = 0;
    for (auto &vec : params)
    {
        total_size += vec.size();
    }
    parameters.reserve(total_size);

    for (auto &vec : params)
    {
        parameters.insert(parameters.end(), vec.begin(), vec.end());
    }
}

void SGD::step()
{
    for (Tensor_ptr tensor : parameters)
    {
        for (int i = 0; i < tensor->total_count; ++i) {
            tensor->at(i) -= lr * tensor->grad_at(i);
        }
    }
}
