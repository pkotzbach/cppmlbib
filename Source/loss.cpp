#include "loss.hpp"
#include <stdexcept>

Tensor_ptr MSELoss(Tensor_ptr input, Tensor_ptr target)
{
    if (input->shape != target->shape) throw std::invalid_argument("input and target must have same shape");
    if (input->shape.size() != 2) throw std::invalid_argument("invalid input size");
    Tensor_ptr result = Tensor::init(input->shape, true);
    result = input - target;
    result = result * result;
    result = result->sum();

    Tensor_ptr div = Tensor::init({1}, std::vector<double>{static_cast<double>(input->shape[0])});
    result = result / div;
    return result;
}
