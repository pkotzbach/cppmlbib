#pragma once

#include "tensor.hpp"

struct Optimizer
{
    Optimizer(std::vector<std::vector<Tensor*>> params, double lr);
    Optimizer(std::vector<Tensor*> params, double lr) : parameters(params), lr(lr) {};
    std::vector<Tensor*> parameters;
    double lr;

    virtual void step() = 0;
    void zero_grad();
};

struct SGD : Optimizer
{
    SGD(std::vector<std::vector<Tensor*>> params, double lr) : Optimizer{params, lr} {}
    SGD(std::vector<Tensor*> params, double lr) : Optimizer{params, lr} {}
    void step() override;
};