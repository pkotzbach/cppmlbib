#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <span>
#include <unordered_set>

struct Tensor;
typedef std::shared_ptr<Tensor> Tensor_ptr;

struct Tensor : std::enable_shared_from_this<Tensor>
{
    static Tensor_ptr init(std::vector<int> shape, bool init_zero = false, std::string device = "cpu");
    static Tensor_ptr init(std::vector<int> shape, std::vector<double> values, std::string device = "cpu");
    // static Tensor_ptr init(std::vector<int> shape, std::vector<double> values, std::vector<double> grads, std::string device = "cpu") {init_internal(shape, values, grads, false, device);}

    Tensor& init_internal(std::vector<int> shape, std::vector<double> init_values, std::vector<double> init_grads, bool init_zero, std::string device);
    Tensor() {}
    ~Tensor();
    
private:
    double* values;
    double* grads;
public:
    std::vector<double> values_vec();
    std::vector<double> grads_vec();

    std::vector<int> strides;
    std::vector<int> shape;
    int total_count;

    std::pair<Tensor_ptr, Tensor_ptr> parents;
    std::function<void()> backward_fn;
    std::string op;
    std::string device;

    // TODO: smaller API, span?
    double& _at(std::vector<int> idx, double* source);
    double& _at(int flat_idx, double* source);

    double& at(std::vector<int> idx) { return _at(idx, values); }
    double& at(int flat_idx)       { return _at(flat_idx, values); }

    double& grad_at(std::vector<int> idx) { return _at(idx, grads); }
    double& grad_at(int flat_idx)       { return _at(flat_idx, grads); }


    Tensor_ptr sum();
    Tensor_ptr sum(int axis);
    Tensor_ptr relu();
    Tensor_ptr exp();
    Tensor_ptr argmax(int axis = 1);
    Tensor_ptr max(int axis);
    Tensor_ptr matmul(Tensor_ptr tensor);
    Tensor_ptr transpose();

    Tensor& flatten();
    
    // autograd
    void backward();
    void zero_grad();
    void toposort(Tensor_ptr t, std::unordered_set<Tensor_ptr>& visited, std::vector<Tensor_ptr>& res);

    // Tensor_ptr operator=(Tensor_ptr tensor);
};

// TODO: put it inside struct
Tensor_ptr operator+(Tensor_ptr self, Tensor_ptr other);
Tensor_ptr operator-(Tensor_ptr self, Tensor_ptr other);
Tensor_ptr operator*(Tensor_ptr self, Tensor_ptr other);
Tensor_ptr operator/(Tensor_ptr self, Tensor_ptr other);
