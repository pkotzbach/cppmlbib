#pragma once

#include "value.hpp"
#include "tensor.hpp"
#include <stdexcept>

struct Component {
    virtual Tensor forward(Tensor input) {
        throw std::runtime_error("forward not implemented");
    }
};

struct Linear : Component
{
    Tensor biases;
    Tensor weights;

    int in_size, out_size;

    Linear(int in_size, int out_size);
    ~Linear() {}
    Tensor forward(Tensor input) override;
    std::vector<Tensor*> params() {return {&weights, &biases};}
};

struct Softmax : Component
{
    Softmax() {};
    Tensor forward(Tensor input) override;
};
