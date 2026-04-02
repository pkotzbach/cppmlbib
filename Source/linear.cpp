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

Convolution::Convolution(int in_channels, int i_out_channels, int i_kernel_size, int i_stride, int i_padding, std::string device) 
{
    weights = Tensor::init({i_kernel_size * i_kernel_size * in_channels, i_out_channels});
    biases = Tensor::init({1, i_out_channels});

    stride = i_stride;
    kernel_size = i_kernel_size;
    padding = i_padding;
    out_channels = i_out_channels;
}

Tensor_ptr Convolution::forward(Tensor_ptr input) {
    if (input->get_shape().size() != 4) throw std::invalid_argument("Invalid input size");
    int height = input->get_shape(2);
    int width = input->get_shape(3);
    int out_h = (height - kernel_size + 2 * padding) / stride + 1;
    int out_w = (width  - kernel_size + 2 * padding) / stride + 1;

    Tensor_ptr folded = input->im2col(kernel_size, stride, padding);
    int batches = folded->get_shape(0);
    int columns = folded->get_shape(1); // = out_h * out_w
    int kernel = folded->get_shape(2);
    folded = folded->view({batches * columns, kernel});

    Tensor_ptr result = folded->matmul(weights) + biases;
    // matmul_res shape = [batches * columns, out_channels]
    result = result->view({batches, out_channels, out_h, out_w});
    return result;
}