#pragma once

#include "value.hpp"
#include "tensor.hpp"
#include <stdexcept>

struct Component
{
    std::string device;
    virtual Tensor forward(Tensor input)
    {
        throw std::runtime_error("forward not implemented");
    }
};

struct Linear : Component
{
    int in_size, out_size;

    Tensor weights;
    Tensor biases;

    Linear(int in_size, int out_size, std::string device = "cpu") : in_size(in_size),
                                                                    out_size(out_size), weights({out_size, in_size}, false, device), 
                                                                    biases({out_size}, false, device) { this->device = device; }
    ~Linear() {}
    Tensor forward(Tensor input) override;
    std::vector<Tensor*> params() { return {&weights, &biases}; }
};

struct Softmax : Component
{
    Softmax() {};
    Tensor forward(Tensor input) override;
};
