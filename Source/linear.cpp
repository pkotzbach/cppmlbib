#include "linear.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

Tensor_ptr Linear::forward(Tensor_ptr input) {
    if (input->get_shape().size() != 2 && input->get_shape(1) != in_size) throw std::invalid_argument("Invalid input size");
    return input->matmul(weights) + biases;
}

Tensor_ptr Softmax::forward(Tensor_ptr input) {
    return input->softmax();
}

Convolution::Convolution(int in_channels, int kernel_size, int out_channels, std::string device)
{
    weights = Tensor::init({kernel_size * kernel_size * in_channels, kernel_size * kernel_size * out_channels});
    biases = Tensor::init({1, kernel_size * kernel_size * out_channels});
}

Tensor_ptr Convolution::forward(Tensor_ptr input) {
    if (input->get_shape().size() != 2) throw std::invalid_argument("Invalid input size");
    return input->matmul(weights) + biases;
}