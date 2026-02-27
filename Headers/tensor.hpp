#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <span>
#include <unordered_set>

class Tensor;
typedef std::shared_ptr<Tensor> Tensor_ptr;

class Tensor : public std::enable_shared_from_this<Tensor>
{
private:
    std::shared_ptr<double[]> values;
    std::shared_ptr<double[]> grads;

    std::vector<int> strides;
    std::pair<Tensor_ptr, Tensor_ptr> parents;
    std::function<void()> backward_fn;
    std::string op;

    std::vector<int> shape;
    std::string device;
    int total_count;

    std::vector<int> broadcast_strides(int ndim);
    int strided_idx(int shape_idx, std::vector<int>& strides, std::vector<int>& shape);
    int strided_idx(std::vector<int> idx);

    Tensor& init_internal(std::vector<int> shape, std::vector<double> init_values, std::vector<double> init_grads, bool init_zero, std::string device);

public:
    static Tensor_ptr init(std::vector<int> shape, bool init_zero = false, std::string device = "cpu");
    static Tensor_ptr init(std::vector<int> shape, std::vector<double> values, std::string device = "cpu");
    // static Tensor_ptr init(std::vector<int> shape, std::vector<double> values, std::vector<double> grads, std::string device = "cpu") {init_internal(shape, values, grads, false, device);}

    Tensor() {}
    ~Tensor() {};

    // getters
    std::vector<int>& get_shape() {return shape;}
    int get_shape(int idx) {return shape[idx];}
    int get_total_count() {return total_count;}
    std::string get_device() {return device;}

    std::vector<double> values_vec(int count, std::vector<int>& strides, std::vector<int>& shape);
    std::vector<double> values_vec() {return values_vec(total_count, strides, shape);};
    std::vector<double> grads_vec();

    // TODO: smaller API, span?
    double& at(std::vector<int> shape_idx) { return values[strided_idx(shape_idx)]; }
    double& at(int shape_idx)              { return values[strided_idx(shape_idx, strides, shape)]; }
    double& at(int shape_idx, std::vector<int>& strides, std::vector<int>& shape) {
        return values[strided_idx(shape_idx, strides, shape)]; }

    double& grad_at(std::vector<int> shape_idx) { return grads[strided_idx(shape_idx)]; }
    double& grad_at(int shape_idx)              { return grads[strided_idx(shape_idx, strides, shape)]; }
    double& grad_at(int shape_idx, std::vector<int> strides, std::vector<int> shape) {
        return grads[strided_idx(shape_idx, strides, shape)]; }

    // operators (grad)
    Tensor_ptr sum();
    Tensor_ptr sum(int axis);
    Tensor_ptr relu();
    Tensor_ptr exp();
    // Tensor_ptr max(int axis);
    Tensor_ptr matmul(Tensor_ptr tensor);
    Tensor_ptr softmax();
    // simple operators
    friend Tensor_ptr operator+(Tensor_ptr self, Tensor_ptr other);
    friend Tensor_ptr operator-(Tensor_ptr self, Tensor_ptr other);
    friend Tensor_ptr operator*(Tensor_ptr self, Tensor_ptr other);
    friend Tensor_ptr operator/(Tensor_ptr self, Tensor_ptr other);
    
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

// TODO: not here
std::vector<int> broadcast_shape(std::vector<int>& a, std::vector<int>& b);