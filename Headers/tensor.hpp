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
    ~Tensor() {};
    
private:
    std::shared_ptr<double[]> values;
    std::shared_ptr<double[]> grads;
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
    int strided_idx(int shape_idx, std::vector<int>& strides, std::vector<int>& shape);
    int strided_idx(std::vector<int> idx);

    double& at(std::vector<int> shape_idx) { return values[strided_idx(shape_idx)]; }
    double& at(int shape_idx)              { return values[strided_idx(shape_idx, strides, shape)]; }
    double& at(int shape_idx, std::vector<int>& strides, std::vector<int>& shape) {
        return values[strided_idx(shape_idx, strides, shape)]; }

    double& grad_at(std::vector<int> shape_idx) { return grads[strided_idx(shape_idx)]; }
    double& grad_at(int shape_idx)              { return grads[strided_idx(shape_idx, strides, shape)]; }
    double& grad_at(int shape_idx, std::vector<int> strides, std::vector<int> shape) {
        return grads[strided_idx(shape_idx, strides, shape)]; }

    std::vector<int> broadcast_strides(int ndim);

    // operators (grad)
    Tensor_ptr sum();
    Tensor_ptr sum(int axis);
    Tensor_ptr relu();
    Tensor_ptr exp();
    Tensor_ptr max(int axis);
    Tensor_ptr matmul(Tensor_ptr tensor);
    
    // operators (no grad)
    Tensor_ptr argmax(int axis = 1);

    // transformation
    Tensor_ptr transpose();

    // autograd
    void backward();
    void zero_grad();
    void toposort(Tensor_ptr t, std::unordered_set<Tensor_ptr>& visited, std::vector<Tensor_ptr>& res);

    // Tensor_ptr operator=(Tensor_ptr tensor);
};

std::vector<int> broadcast_shape(std::vector<int>& a, std::vector<int>& b);

// TODO: put it inside struct
Tensor_ptr operator+(Tensor_ptr self, Tensor_ptr other);
Tensor_ptr operator-(Tensor_ptr self, Tensor_ptr other);
Tensor_ptr operator*(Tensor_ptr self, Tensor_ptr other);
Tensor_ptr operator/(Tensor_ptr self, Tensor_ptr other);
