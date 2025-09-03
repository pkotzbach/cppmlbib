#pragma once

#include "value.hpp"

struct TensorProxy;
struct Tensor
{
    Tensor& init(std::vector<int> shape);
    Tensor(std::vector<int> shape) {init(shape);}
    Tensor() {}

    std::vector<Value_ptr> data;
    std::vector<int> shape;
    int total_count;

    TensorProxy operator[](int idx);
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