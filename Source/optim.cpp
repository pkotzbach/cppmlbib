#include "optim.hpp"
#include <stdio.h>

void Optimizer::zero_grad()
{
    for (Tensor *tensor : parameters)
    {
        for (int i = 0; i < tensor->total_count; ++i)
            tensor->values[i]->grad = 0;
    }
}

Optimizer::Optimizer(std::vector<std::vector<Tensor *>> params, double lr) : lr(lr)
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
    for (Tensor *tensor : parameters)
    {
        for (int i = 0; i < tensor->total_count; ++i) {
            tensor->values[i]->data -= lr * tensor->values[i]->grad;
        }
    }
}
