#pragma once
#include <functional>
#include <memory>

struct Value : std::enable_shared_from_this<Value>
{
    double data;
    double grad;
    char name;
    std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>> parents;
    std::function<void(Value*)> backward_fn;

    Value(double data=0, std::pair<Value*, Value*> parents = std::pair{nullptr, nullptr}, char name='0') : data(data), parents(parents) {
        grad = 0.0f;
        name = name;
    }

    //operations
    Value operator+(std::shared_ptr<Value>& other);
    // Value operator-(Value& other);
    Value operator*(Value& other);
    Value operator/(Value& other);
    Value relu();
    Value exp();

    void backward();
    void zero_grad();
    void toposort(Value* v, std::vector<Value*>& visited, std::vector<Value*>& res);
};

typedef std::shared_ptr<Value> Value_ptr;