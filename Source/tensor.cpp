#include "tensor.hpp"

#include <stdexcept>
#include <cassert>

Tensor &Tensor::init(std::vector<int> shape)
{
    if (shape.size() == 0)
        throw std::runtime_error("shape 0");
    this->shape = shape;

    total_count = shape[0];
    for (size_t i = 1; i < shape.size(); ++i)
    {
        if (shape[i] <= 0)
            throw std::runtime_error("shape <= 0");
        total_count *= shape[i];
    }

    data.resize(total_count);
    for (int i = 0; i < total_count; ++i)
    {
        data[i] = std::make_shared<Value>(0.0); // init val?
    }
    return *this;
}

//------------------ proxy

TensorProxy Tensor::operator[](int idx)
{
    if (idx < 0 || idx >= shape[0])
    {
        throw std::out_of_range("index out of range");
    }
    int offset = idx * total_count / shape[0];
    return TensorProxy(*this, offset, total_count / shape[0], 1);
}

TensorProxy TensorProxy::operator[](int idx)
{
    if (idx < 0 || idx >= tensor.shape[dim])
    {
        throw std::out_of_range("index out of range");
    }
    int new_offset = offset + idx * proxy_count / tensor.shape[dim];
    return TensorProxy(tensor, new_offset, proxy_count / tensor.shape[dim], dim + 1);
}

Value_ptr TensorProxy::get()
{
    if (dim != tensor.shape.size())
    {
        throw std::invalid_argument("get only allowed at the last dimension");
    }
    return tensor.data[offset];
}

Value_ptr TensorProxy::operator->() {
    return get();
}

TensorProxy::operator Value_ptr &()
{
    return tensor.data[offset];
}

TensorProxy &TensorProxy::operator=(const double &value)
{
    if (dim != tensor.shape.size())
    {
        throw std::invalid_argument("Assignment only allowed at the last dimension");
    }
    Value_ptr new_val = std::make_shared<Value>(value);
    tensor.data[offset] = new_val;
    return *this;
}

TensorProxy &TensorProxy::operator=(Value_ptr value)
{
    if (dim != tensor.shape.size())
    {
        throw std::invalid_argument("Assignment only allowed at the last dimension");
    }
    tensor.data[offset] = value;
    return *this;
}
