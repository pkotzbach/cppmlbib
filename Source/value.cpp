#include <string>
#include <iostream>
#include <functional>
#include <cmath>
#include <memory>

#include "value.hpp"

Value_ptr operator+(Value_ptr self, Value_ptr other)
{
    Value_ptr out = std::make_shared<Value>(self->data + other->data, std::pair{self, other}, '+');
    out->backward_fn = [](Value *self)
    {
        self->parents.first->grad += self->grad;
        self->parents.second->grad += self->grad;
    };
    return out;
}

Value_ptr operator-(Value_ptr self, Value_ptr other)
{
    Value_ptr out = std::make_shared<Value>(self->data - other->data, std::pair{self, other}, '-');
    out->backward_fn = [](Value *self)
    {
        self->parents.first->grad += self->grad;
        self->parents.second->grad -= self->grad;
    };
    return out;
}

Value_ptr operator*(Value_ptr self, Value_ptr other)
{
    Value_ptr out = std::make_shared<Value>(self->data * other->data, std::pair{self, other}, '*');
    out->backward_fn = [](Value *self)
    {
        self->parents.first->grad += self->grad * self->parents.second->data;
        self->parents.second->grad += self->grad * self->parents.first->data;
    };
    return out;
}

Value_ptr operator/(Value_ptr self, Value_ptr other)
{
    Value_ptr out = std::make_shared<Value>(self->data / other->data, std::pair{self, other}, '/');
    out->backward_fn = [](Value *self)
    {
        self->parents.first->grad += self->grad * (1 / self->parents.second->data);
        self->parents.second->grad += self->grad * -(self->parents.first->data / (self->parents.second->data * self->parents.second->data));
    };
    return out;
}

Value_ptr Value::relu()
{
    Value_ptr out = std::make_shared<Value>(data > 0 ? data : 0, std::pair{shared_from_this(), nullptr}, 'r');
    out->backward_fn = [](Value *self)
    {
        self->parents.first->grad += (self->data > 0 ? 1 : 0) * self->grad;
    };
    return out;
}

Value_ptr Value::exp()
{
    Value_ptr out = std::make_shared<Value>(std::exp(data), std::pair{shared_from_this(), nullptr}, 'e');
    out->backward_fn = [](Value *self)
    {
        self->parents.first->grad += self->grad * self->data;
    };
    return out;
}

void Value::toposort(Value *v, std::vector<Value *> &visited, std::vector<Value *> &res)
{
    if (std::find(visited.begin(), visited.end(), v) == visited.end())
    {
        visited.push_back(v);
        if (v->parents.first)
            toposort(v->parents.first.get(), visited, res);
        if (v->parents.second)
            toposort(v->parents.second.get(), visited, res);
        res.push_back(v);
    }
}

void Value::backward()
{
    std::vector<Value *> visited{};
    std::vector<Value *> topo{};
    toposort(this, visited, topo);
    std::reverse(topo.begin(), topo.end());
    grad = 1.0f;
    for (auto *val : topo)
    {
        if (val->backward_fn)
        {
            // printf("grad %c before %f\n", val->op, val->grad);
            val->backward_fn(val);
            // printf("grad %c after %f\n", val->op, val->grad);
        }
    }
}

void Value::zero_grad()
{
    grad = 0;
    if (parents.first && parents.first.get() != this)   parents.first->zero_grad();
    if (parents.second && parents.second.get() != this) parents.second->zero_grad();
}