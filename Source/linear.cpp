#include "linear.hpp"
#include <cstdio>

Linear::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size)
{
    biases = new Value[out_size];
    weights = new Value *[out_size];

    for (int x = 0; x < out_size; ++x)
    {
        weights[x] = new Value[in_size];
    }
}

Linear::~Linear()
{
    for (int x = 0; x < out_size; ++x)
    {
        delete[] weights[x];
    }

    delete[] weights;
}

Value *Linear::forward(Value *input)
{
    Value *output = new Value[out_size];

    for (int x = 0; x < out_size; ++x)
    {
        output[x] = biases[x];
        for (int y = 0; y < in_size; ++y)
        {
            Value temp = input[y] * weights[x][y];
            output[x] = output[x] + temp;
        }
    }

    return output;
}

Value *Softmax::forward(Value *input)
{
    val_sum = Value();
    Value *output = new Value[size];
    for (int x = 0; x < size; ++x)
    {
        output[x] = input[x].exp();
        val_sum = val_sum + output[x];
        printf("%p %p\n", val_sum.parents.first, val_sum.parents.second);
    }

    for (int x = 0; x < size; ++x)
    {
        output[x] = output[x] / val_sum;
    }
    return output;
}