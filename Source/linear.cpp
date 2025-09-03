#include "linear.hpp"
#include <cstdio>

Linear::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size)
{
    weights = Tensor({in_size, out_size});
    biases = Tensor({out_size});
}

Tensor Linear::forward(Tensor input)
{
    Tensor output({out_size});

    for (int x = 0; x < out_size; ++x)
    {
        for (int y = 0; y < in_size; ++y)
        {
            output[x] = output[x].get() + input[y].get() * weights[x][y].get();
        }
        output[x] = output[x].get() + biases[x].get();
    }

    return output;
}

Softmax::Softmax(int size) : size(size)
{
    val_sum = std::make_shared<Value>(0);
}

Tensor Softmax::forward(Tensor input)
{
    Tensor output({size});
    for (int x = 0; x < size; ++x)
    {
        output[x] = input[x].get()->exp();
        val_sum = val_sum + output[x].get();
    }

    for (int x = 0; x < size; ++x)
    {
        output[x] = output[x].get() / val_sum;
    }
    return output;
}