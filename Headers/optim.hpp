#pragma once

#include "tensor.hpp"

class Optimizer {
protected:
    std::vector<Tensor_ptr> parameters;
    float lr;

public:
    Optimizer(std::vector<std::vector<Tensor_ptr>> params, float lr);
    Optimizer(std::vector<Tensor_ptr> params, float lr) : parameters(std::move(params)), lr(lr) {};

    virtual void step() = 0;
    void zero_grad();
};

class SGD : public Optimizer {
public:
    SGD(std::vector<std::vector<Tensor_ptr>> params, float lr) : Optimizer{params, lr} {}
    SGD(std::vector<Tensor_ptr> params, float lr) : Optimizer{params, lr} {}
    void step() override;
};