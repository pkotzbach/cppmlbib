#include "tensor.hpp"

#include <stdexcept>
#include <cassert>
#include <random>

double sample_kaiming(double n)
{
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, std::sqrt(2/n));
    return dist(gen);
}

Tensor &Tensor::init(std::vector<int> shape, bool init_zero)
{
    if (shape.size() == 0) throw std::runtime_error("shape 0");
    this->shape = shape;

    total_count = shape[0];
    for (size_t i = 1; i < shape.size(); ++i)
    {
        if (shape[i] <= 0) throw std::runtime_error("shape <= 0");
        total_count *= shape[i];
    }

    values.resize(total_count);
    for (int i = 0; i < total_count; ++i)
    {
        values[i] = std::make_shared<Value>(init_zero? 0: sample_kaiming(total_count)); // init val?
    }
    return *this;
}

Tensor Tensor::relu()
{
    Tensor result(shape);
    for (int i = 0; i < total_count; ++i) {
        result[i] = values[i]->relu();
    }
    return result;
}

Tensor &Tensor::flatten()
{
    shape = {total_count};
    return *this;
}

//tensor operators
Tensor Tensor::operator+(Tensor tensor)
{
    if (shape != tensor.shape) throw std::invalid_argument("Shape must be the same");
    Tensor result(shape, true);
    for (int i = 0; i < total_count; ++i) {
        result.values[i] = values[i] + tensor.values[i];
    }
    return result;
}

Tensor Tensor::operator-(Tensor tensor)
{
    if (shape != tensor.shape) throw std::invalid_argument("Shape must be the same");
    Tensor result(shape, true);
    for (int i = 0; i < total_count; ++i) {
        result.values[i] = values[i] - tensor.values[i];
    }
    return result;
}

Tensor Tensor::operator*(Tensor tensor)
{
    if (shape != tensor.shape) throw std::invalid_argument("Shape must be the same");
    Tensor result(shape, true);
    for (int i = 0; i < total_count; ++i) {
        result.values[i] = values[i] * tensor.values[i];
    }
    return result;
}

Tensor &Tensor::operator=(Tensor tensor)
{
    values = tensor.values;
    shape = tensor.shape;
    total_count = tensor.total_count;
    return *this;
}

//------------------ proxy

TensorProxy Tensor::operator[](int idx)
{
    if (idx < 0 || idx >= shape[0]) throw std::out_of_range("index out of range");
    int offset = idx * total_count / shape[0];
    return TensorProxy(*this, offset, total_count / shape[0], 1);
}

TensorProxy TensorProxy::operator[](int idx)
{
    if (idx < 0 || idx >= tensor.shape[dim]) throw std::out_of_range("index out of range");
    int new_offset = offset + idx * proxy_count / tensor.shape[dim];
    return TensorProxy(tensor, new_offset, proxy_count / tensor.shape[dim], dim + 1);
}

Value_ptr TensorProxy::get()
{
    if (dim != tensor.shape.size()) throw std::invalid_argument("get only allowed at the last dimension");
    return tensor.values[offset];
}

Value_ptr TensorProxy::operator->() {
    return get();
}

TensorProxy::operator Value_ptr &()
{
    return tensor.values[offset];
}

TensorProxy &TensorProxy::operator=(const double &value)
{
    if (dim != tensor.shape.size()) throw std::invalid_argument("Assignment only allowed at the last dimension");
    Value_ptr new_val = std::make_shared<Value>(value);
    tensor.values[offset] = new_val;
    return *this;
}

TensorProxy &TensorProxy::operator=(Value_ptr value)
{
    if (dim != tensor.shape.size()) throw std::invalid_argument("Assignment only allowed at the last dimension");
    tensor.values[offset] = value;
    return *this;
}
