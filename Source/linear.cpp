#include "linear.hpp"
#include <cstdio>

Linear::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size)
{
    weights = Tensor({out_size, in_size});
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

Value_ptr MSELoss(Tensor input, Tensor target)
{
    if (input.shape != target.shape) throw std::invalid_argument("input and target must have same shape");
    if (input.shape.size() != 1) throw std::invalid_argument("shape must have dim 1");
    Tensor temp(input.shape);
    temp = input - target;
    temp = temp * temp;

    Value_ptr result = std::make_shared<Value>(0);
    for (int i = 0; i < temp.data.size(); ++i)
        result = result + temp.data[i];

    Value_ptr div = std::make_shared<Value>(temp.data.size());
    result = result / div;

    return result;
}
