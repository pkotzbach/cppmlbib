#pragma once

#include "value.hpp"

struct Linear
{
    Value** weights;
    Value* biases;

    int in_size, out_size;

    Linear(int in_size, int out_size);
    ~Linear();
    Value* forward(Value* input);
};

struct Softmax
{
    int size;
    Value val_sum;

    Softmax(int size) : size(size) {}
    Value* forward(Value* input);
};