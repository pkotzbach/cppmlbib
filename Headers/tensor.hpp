#pragma once

#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <span>
#include <unordered_set>
#include "globals.hpp"
#include "stride.hpp"

class Tensor;
struct BinaryOpContext;
typedef std::shared_ptr<Tensor> Tensor_ptr;

struct Storage {
    // TODO: should be aligned?
    std::shared_ptr<float[]> data;
    int size; // TODO: duplication of total_count
    std::string device;

    Storage() {}
    Storage(std::string device, std::vector<float> values, int size);

    float* get() { return data.get(); }
    void set(int idx, float val);
    float at(int idx);
    float& operator[](int idx);

    std::vector<float> cpu();
};

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    Storage values;
    Storage grads;

    std::pair<Tensor_ptr, Tensor_ptr> parents;
    std::function<void()> backward_fn;
    std::string op;
    
    std::vector<int> strides;
    std::vector<int> shape;
    int total_count;
    std::string device;

    Tensor& init_internal(std::vector<int> shape, std::vector<float> init_values, std::vector<float> init_grads, bool init_zero, std::string device);

public:
    static Tensor_ptr init(std::vector<int> shape, bool init_zero = false, std::string device = "cpu");
    static Tensor_ptr init(std::vector<int> shape, std::vector<float> values, std::string device = "cpu");
    // static Tensor_ptr init(std::vector<int> shape, std::vector<float> values, std::vector<float> grads, std::string device = "cpu") {init_internal(shape, values, grads, false, device);}

    Tensor() {}
    ~Tensor() {}

    //// getters setters
    std::vector<int>& get_shape() {return shape;}
    int get_shape(int idx) {return shape[idx];}
    void set_shape(std::vector<int> shp) { 
        shape = shp; 
        total_count = 1;
        for (auto s : shape) total_count *= s;
    }

    std::vector<int>& get_strides() {return strides;}
    void set_strides(std::vector<int> str) { strides = str; }

    int get_total_count() {return total_count;}
    std::string get_device() {return device;}

    // storage
    std::vector<float> values_vec(int count, std::vector<int>& strides, std::vector<int>& shape);
    std::vector<float> values_vec() {return values_vec(total_count, strides, shape); }
    std::vector<float> grads_vec();
    float* raw_values() { return values.get(); }
    float* raw_grads() { return grads.get(); }

    void set(std::vector<int> indicies, float val) { values.set(stride::strided_idx(indicies, strides, shape), val); }
    void set(int shape_idx, float val) { values.set(stride::strided_idx(shape_idx, strides, shape), val); }
    void set(int shape_idx, const std::vector<int>& strides, const std::vector<int>& shape, float val) {
        values.set(stride::strided_idx(shape_idx, strides, shape), val); }

    float get(std::vector<int> indicies) { return values.at(stride::strided_idx(indicies, strides, shape)); }
    float get(int shape_idx)              { return values.at(stride::strided_idx(shape_idx, strides, shape)); }
    float get(int shape_idx, const std::vector<int>& strides, const std::vector<int>& shape) {
        return values.at(stride::strided_idx(shape_idx, strides, shape)); }

    void grad_set(std::vector<int> shape_idx, float val) { grads.set(stride::strided_idx(shape_idx, strides, shape), val); }
    void grad_set(int shape_idx, float val)              { grads.set(stride::strided_idx(shape_idx, strides, shape), val); }
    void grad_set(int shape_idx, std::vector<int> strides, std::vector<int> shape, float val) {
        grads.set(stride::strided_idx(shape_idx, strides, shape), val); }

    float grad_get(std::vector<int> shape_idx) { return grads.at(stride::strided_idx(shape_idx, strides, shape)); }
    float grad_get(int shape_idx)              { return grads.at(stride::strided_idx(shape_idx, strides, shape)); }
    float grad_get(int shape_idx, std::vector<int> strides, std::vector<int> shape) {
        return grads.at(stride::strided_idx(shape_idx, strides, shape)); }

    //// operators (grad)
    Tensor_ptr relu();
    Tensor_ptr exp();
    Tensor_ptr matmul(Tensor_ptr tensor);
    Tensor_ptr softmax();
    //// binary operations
    friend BinaryOpContext;
    friend Tensor_ptr operator+(Tensor_ptr self, Tensor_ptr other);
    friend Tensor_ptr operator-(Tensor_ptr self, Tensor_ptr other);
    friend Tensor_ptr operator*(Tensor_ptr self, Tensor_ptr other);
    friend Tensor_ptr operator/(Tensor_ptr self, Tensor_ptr other);
    
    //// reduction
    // no grad (for sure?)
    Tensor_ptr argmax(int axis = 1);
    Tensor_ptr max();
    // grad
    Tensor_ptr sum();
    Tensor_ptr sum(int axis);

    //// transformation
    Tensor_ptr transpose();

    //// autograd
    void backward();
    void zero_grad();
    void toposort(Tensor_ptr t, std::unordered_set<Tensor_ptr>& visited, std::vector<Tensor_ptr>& res);

    // Tensor_ptr operator=(Tensor_ptr tensor);
};
