#include "loss.hpp"
#include <stdexcept>

Value_ptr MSELoss(Tensor input, Tensor target)
{
    if (input.shape != target.shape) throw std::invalid_argument("input and target must have same shape");
    if (input.shape.size() != 2) throw std::invalid_argument("invalid input size");
    Tensor temp(input.shape, true);
    temp = input - target;
    temp = temp * temp;

    Value_ptr result = std::make_shared<Value>(0);
    for (int i = 0; i < temp.total_count; ++i)
        result = result + temp.values[i];

    Value_ptr div = std::make_shared<Value>(temp.total_count);
    result = result / div;

    return result;
}
