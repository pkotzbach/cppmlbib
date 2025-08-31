#include <string>
#include <iostream>
#include <functional>
#include <cmath>
#include <memory>

#include "value.hpp"

Value Value::operator+(Value_ptr& other)
{
    Value_ptr out = std::make_shared<Value>(data + other->data, std::pair{shared_from_this(), other});
    out->backward_fn = [](Value *self)
    {
        self->parents.first->grad += self->grad;
        self->parents.second->grad += self->grad;
    };
    return *out;
}

Value Value::operator*(Value &other)
{
    Value out(data * other.data, std::pair{this, &other});
    out.backward_fn = [](Value *self)
    {
        self->parents.first->grad += self->grad * self->parents.second->data;
        self->parents.second->grad += self->grad * self->parents.first->data;
    };
    return out;
}

Value Value::operator/(Value &other)
{
    Value out(data / other.data, std::pair{this, &other});
    out.backward_fn = [](Value *self)
    {
        self->parents.first->grad += self->grad * (1 / self->parents.second->data);
        self->parents.second->grad += self->grad * -(self->parents.first->data / (self->parents.second->data * self->parents.second->data));
    };
    return out;
}

Value Value::relu()
{
    Value out(data > 0 ? data : 0, std::pair{this, nullptr});
    out.backward_fn = [](Value *self)
    {
        self->parents.first->grad += (self->data > 0 ? 1 : 0) * self->grad;
    };
    return out;
}

Value Value::exp()
{
    Value out(std::exp(data), std::pair{this, nullptr});
    out.backward_fn = [](Value *self)
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
            toposort(v->parents.first, visited, res);
        if (v->parents.second)
            toposort(v->parents.second, visited, res);
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
            // printf("grad %c before %f\n", val->name, val->grad);
            val->backward_fn(val);
            // printf("grad %c after %f\n", val->name, val->grad);
        }
    }
}

void Value::zero_grad()
{
    grad = 0;
    if (parents.first && parents.first != this)   parents.first->zero_grad();
    if (parents.second && parents.second != this) parents.second->zero_grad();
}