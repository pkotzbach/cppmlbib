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
    if (input->get_shape().size() != 2) throw std::invalid_argument("Invalid input size");
    Tensor_ptr exps = input->exp();
    Tensor_ptr sums = exps->sum(1);

    Tensor_ptr result = Tensor::init(input->get_shape());
    for (int n = 0; n < input->get_shape(0); ++n)
        for (int c = 0; c < input->get_shape(1); ++c)
            result->at({n, c}) = exps->at({n, c}) / sums->at({n});
    
    return result;
}
