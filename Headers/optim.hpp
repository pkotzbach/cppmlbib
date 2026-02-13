#pragma once

#include "tensor.hpp"

class Optimizer
{
protected:
    std::vector<Tensor_ptr> parameters;
    double lr;

public:
    Optimizer(std::vector<std::vector<Tensor_ptr>> params, double lr);
    Optimizer(std::vector<Tensor_ptr> params, double lr) : parameters(params), lr(lr) {};

    virtual void step() = 0;
    void zero_grad();
};

class SGD : public Optimizer
{
public:
    SGD(std::vector<std::vector<Tensor_ptr>> params, double lr) : Optimizer{params, lr} {}
    SGD(std::vector<Tensor_ptr> params, double lr) : Optimizer{params, lr} {}
    void step() override;
};