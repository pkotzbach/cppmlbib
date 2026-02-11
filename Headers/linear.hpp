#pragma once

#include "tensor.hpp"
#include <stdexcept>

struct Component
{
    std::string device;
    virtual Tensor_ptr forward(Tensor_ptr input)
    {
        throw std::runtime_error("forward not implemented");
    }
};

struct Linear : Component
{
    int in_size, out_size;

    Tensor_ptr weights;
    Tensor_ptr biases;

    Linear(int in_size, int out_size, std::string device = "cpu") : in_size(in_size),
                                                                    out_size(out_size), weights(Tensor::init({in_size, out_size}, false, device)), 
                                                                    biases(Tensor::init({1, out_size}, false, device)) { this->device = device; }
    ~Linear() {}
    Tensor_ptr forward(Tensor_ptr input) override;
    std::vector<Tensor_ptr> params() { return {weights, biases}; }
};

struct Softmax : Component
{
    Softmax() {};
    Tensor_ptr forward(Tensor_ptr input) override;
};
