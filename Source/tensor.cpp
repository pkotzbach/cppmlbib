#include "tensor.hpp"

#include <stdexcept>
#include <cassert>
#include <random>
#include <string>

double sample_kaiming(double n)
{
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, std::sqrt(2/n));
    return dist(gen);
}

Tensor &Tensor::init(std::vector<int> shape, std::vector<double> init_values, bool init_zero, std::string device)
{
    if (shape.size() == 0) throw std::runtime_error("shape 0");
    if (device.compare("cuda") != 0 && device.compare("cpu") != 0) throw std::runtime_error("invalid device!");
    this->shape = shape;
    this->device = device;

    total_count = shape[0];
    for (size_t i = 1; i < shape.size(); ++i)
    {
        if (shape[i] <= 0) throw std::runtime_error("shape <= 0");
        total_count *= shape[i];
    }

    if (init_values.size() > 0 && init_values.size() != total_count) throw std::runtime_error("init values doesnt match shape");

    values.resize(total_count);
    for (int i = 0; i < total_count; ++i)
    {
        if (init_values.size() > 0) values[i] = std::make_shared<Value>(init_values[i]);
        else values[i] = std::make_shared<Value>(init_zero? 0: sample_kaiming(total_count));
    }
    
    return *this;
}

std::vector<double> Tensor::raw_values()
{
    std::vector<double> raw_values;
    for (auto& val : values) {
        raw_values.push_back(val->data);
    }
    return raw_values;
}

Tensor Tensor::relu()
{
    Tensor result(shape);
    for (int i = 0; i < total_count; ++i) {
        result.values[i] = values[i]->relu();
    }
    return result;
}

Tensor Tensor::argmax(int dim)
{
    // assume 2d matrix
    if (dim != 1) throw std::invalid_argument("Only for dim 1 now");
    if (shape.size() != 2) throw std::invalid_argument("Only for 2d matrix now");

    std::vector<int> result_shape;
    for (int i = 0; i < shape.size(); ++i) {
        result_shape.push_back(i == dim ? 1 : shape[i]);
    }
    Tensor result(result_shape);

    int idx = -1;
    double max, data;
    for (int i = 0; i < shape[0]; ++i) {
        max = -INFINITY;
        idx = -1;
        for (int j = 0; j < shape[dim]; ++j) {
            data = (*this)[i][j].get()->data;
            if (data > max) {
                max = data;
                idx = j;
            }
        }
        result[i][0] = idx;
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
    device = tensor.device;
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
