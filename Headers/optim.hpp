#pragma once

#include "tensor.hpp"

class Optimizer {
protected:
    std::vector<Tensor_ptr> parameters;
    float lr;
    Device device;

public:
    Optimizer(std::vector<std::vector<Tensor_ptr>> params, float lr, Device device);
    Optimizer(std::vector<Tensor_ptr> params, float lr, Device device) : parameters(std::move(params)), lr(lr), device(device) {};

    virtual void step() = 0;
    void zero_grad();
};

class SGD : public Optimizer {
public:
    SGD(std::vector<std::vector<Tensor_ptr>> params, float lr, Device device) : Optimizer{params, lr, device} {}
    SGD(std::vector<Tensor_ptr> params, float lr, Device device) : Optimizer{params, lr, device} {}
    void step() override;
};