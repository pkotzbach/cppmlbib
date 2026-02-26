#include "linear.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

Tensor_ptr Linear::forward(Tensor_ptr input)
{
    if (input->get_shape().size() != 2 && input->get_shape(1) != in_size) throw std::invalid_argument("Invalid input size");
    return input->matmul(weights) + biases;
}

Tensor_ptr Softmax::forward(Tensor_ptr input)
{
    return input->softmax();
}
