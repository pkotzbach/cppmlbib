#pragma once
#include <functional>
#include <memory>

struct Value;

typedef std::shared_ptr<Value> Value_ptr;

struct Value : std::enable_shared_from_this<Value>
{
    double data;
    double grad;
    char name;
    std::pair<Value_ptr, Value_ptr> parents;
    std::function<void(Value*)> backward_fn;

    Value(double data=0, std::pair<Value_ptr, Value_ptr> parents = std::pair{nullptr, nullptr}, char name='0') : data(data), parents(parents) {
        grad = 0.0f;
        name = name;
    }

    Value_ptr relu();
    Value_ptr exp();

    void backward();
    void zero_grad();
    void toposort(Value* v, std::vector<Value*>& visited, std::vector<Value*>& res);
};

// operators
Value_ptr operator+(Value_ptr self, Value_ptr other);
Value_ptr operator-(Value_ptr self, Value_ptr other);
Value_ptr operator*(Value_ptr self, Value_ptr other);
Value_ptr operator/(Value_ptr self, Value_ptr other);