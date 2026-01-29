#pragma once

#include "value.hpp"

struct TensorProxy;
struct Tensor
{
    Tensor& init(std::vector<int> shape, bool init_zero = false);
    Tensor(std::vector<int> shape, bool init_zero = false) {init(shape, init_zero);}
    Tensor() {}

    std::vector<Value_ptr> values;
    std::vector<int> shape;
    int total_count;

    Tensor relu();
    Tensor argmax(int dim);

    Tensor& flatten();

    // operators
    TensorProxy operator[](int idx);
    Tensor operator+(Tensor tensor);
    Tensor operator-(Tensor tensor);
    Tensor operator*(Tensor tensor);
    Tensor& operator=(Tensor tensor);
};

struct TensorProxy
{
    TensorProxy(Tensor& tensor, int offset, int proxy_count, size_t dim) : tensor(tensor), offset(offset), proxy_count(proxy_count), dim(dim) {}
    Tensor& tensor;
    int offset;
    int proxy_count;
    size_t dim;

    Value_ptr operator->();
    Value_ptr get();
    TensorProxy operator[](int idx);
    operator Value_ptr&();
    TensorProxy& operator=(const double& value);
    TensorProxy& operator=(Value_ptr value);
};