#pragma once

#include "value.hpp"
#include "tensor.hpp"
#include <stdexcept>

struct Component {
    virtual std::vector<Value_ptr> forward(std::vector<Value_ptr> input) {
        throw std::runtime_error("forward not implemented");
    }
};

struct Linear : Component
{
    std::vector<Value_ptr> biases;
    std::vector<std::vector<Value_ptr>> weights;

    int in_size, out_size;

    Linear(int in_size, int out_size);
    ~Linear() {}
    std::vector<Value_ptr> forward(std::vector<Value_ptr> input) override;
};

struct Softmax : Component
{
    int size;
    Value val_sum;

    Softmax(int size) : size(size) {}
    std::vector<Value_ptr> forward(std::vector<Value_ptr> input) override;
};