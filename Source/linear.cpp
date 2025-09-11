#include "linear.hpp"
#include <cstdio>
#include <stdexcept>

Linear::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size)
{
    weights = Tensor({out_size, in_size});
    biases = Tensor({out_size});
}

Tensor Linear::forward(Tensor input)
{
    if (input.shape.size() != 2 && input.shape[1] != in_size) throw std::invalid_argument("Invalid input size");
    const int batch_size = input.shape[0];
    Tensor output({batch_size, out_size}, true);

    auto& val_input = input.values;
    auto& val_weights = weights.values;
    auto& val_biases = biases.values;
    auto& val_output = output.values;

    for (int i = 0; i < batch_size; ++i) {
        for (int x = 0; x < out_size; ++x)
        {
            for (int y = 0; y < in_size; ++y)
            {
                val_output[i * out_size + x] = val_output[i * out_size + x] + val_input[i * in_size + y] * val_weights[x * in_size + y];
            }
            val_output[i * out_size + x] = val_output[i * out_size + x] + val_biases[x];
        }
    }

    return output;
}

Tensor Softmax::forward(Tensor input)
{
    if (input.shape.size() != 2) throw std::invalid_argument("Invalid input size");
    int size = input.shape[1];
    int entries = input.shape[0];
    Tensor output(input.shape, true);
    auto& val_output = output.values;
    auto& val_input = input.values;
    for (int i = 0; i < entries; ++i) {
        Value_ptr val_sum = std::make_shared<Value>(0);
        for (int x = 0; x < size; ++x)
        {
            val_output[i * size + x] = val_input[i * size + x]->exp();
            val_sum = val_sum + val_output[i * size + x];
        }

        for (int x = 0; x < size; ++x)
        {
            val_output[i * size + x] = val_output[i * size + x] / val_sum;
        }
    }
    return output;
}
