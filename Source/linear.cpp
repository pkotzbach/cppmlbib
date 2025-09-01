#include "linear.hpp"
#include <cstdio>

Linear::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size)
{
    biases.reserve(out_size);
    for (int i = 0; i < out_size; ++i)
    {
        biases.push_back(std::make_shared<Value>(0.0));
    }

    weights.resize(out_size);
    for (int i = 0; i < out_size; ++i)
    {
        weights[i].reserve(in_size);
        for (int j = 0; j < in_size; ++j)
        {
            weights[i].push_back(std::make_shared<Value>(0.0)); // or random init
        }
    }
}

std::vector<Value_ptr> Linear::forward(std::vector<Value_ptr> input)
{
    std::vector<Value_ptr> output;
    output.reserve(out_size);
    for (int i = 0; i < out_size; ++i)
    {
        output.push_back(std::make_shared<Value>(0.0));
    }

    for (int x = 0; x < out_size; ++x)
    {
        for (int y = 0; y < in_size; ++y)
        {
            output[x] = output[x] + input[y] * weights[x][y];
        }
        output[x] = output[x] + biases[x];
    }

    return output;
}

std::vector<Value_ptr> Softmax::forward(std::vector<Value_ptr> input)
{
    std::vector<Value_ptr> output;
    output.reserve(size);
    Value_ptr val_sum = std::make_shared<Value>(0.0);
    for (int x = 0; x < size; ++x)
    {
        output[x] = input[x]->exp();
        val_sum = val_sum + output[x];
    }

    for (int x = 0; x < size; ++x)
    {
        output[x] = output[x] / val_sum;
    }
    return output;
}