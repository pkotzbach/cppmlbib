#include "loss.hpp"
#include <stdexcept>

Tensor_ptr MSELoss(Tensor_ptr input, Tensor_ptr target)
{
    if (input->get_shape() != target->get_shape()) throw std::invalid_argument("input and target must have same shape");
    if (input->get_shape().size() != 2) throw std::invalid_argument("invalid input size");
    Tensor_ptr result = Tensor::init(input->get_shape(), true);
    result = input - target;
    result = result * result;
    result = result->sum();

    Tensor_ptr div = Tensor::init({1}, std::vector<double>{static_cast<double>(input->get_shape(0))}, input->get_device());
    result = result / div;
    return result;
}
