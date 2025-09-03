#pragma once

#include "value.hpp"

struct TensorProxy;
struct Tensor
{
    Tensor(std::vector<int> shape);

    std::vector<Value_ptr> data;
    std::vector<int> shape;
    int total_count;

    TensorProxy operator[](int idx);
};

struct TensorProxy
{
    TensorProxy(Tensor& tensor, int offset, int proxy_count, int dim) : tensor(tensor), offset(offset), proxy_count(proxy_count), dim(dim) {}
    Tensor& tensor;
    int offset;
    int proxy_count;
    int dim;

    TensorProxy operator[](int idx);

    operator Value_ptr&() {
        return tensor.data[offset];
    }
    TensorProxy& operator=(const double& value);
};