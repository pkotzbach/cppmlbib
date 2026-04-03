#pragma once

#include "tensor.hpp"
#include <stdexcept>

class Component {
protected:
    Device device;
public:
    virtual Tensor_ptr forward(Tensor_ptr input) {
        throw std::runtime_error("forward not implemented");
    }
};

class Linear : public Component {
private:
    int in_size, out_size;

    Tensor_ptr weights;
    Tensor_ptr biases;

public:
    Linear(int in_size, int out_size, Device device = Device::CPU) : in_size(in_size),
                                                                    out_size(out_size), weights(Tensor::init({in_size, out_size}, false, device)), 
                                                                    biases(Tensor::init({1, out_size}, false, device)) { this->device = device; }
    ~Linear() {}
    Tensor_ptr forward(Tensor_ptr input) override;
    std::vector<Tensor_ptr> params() { return {weights, biases}; }
    int get_in_size() { return in_size; }
    int get_out_size() { return out_size; }
    Tensor_ptr get_weights() { return weights; }
    Tensor_ptr get_biases() { return biases; }

};

class Convolution : public Component {
private:
    Tensor_ptr weights;
    Tensor_ptr biases;

    int stride;
    int kernel_size;
    int padding;
    int out_channels;

public:
    Convolution(int in_channels, int out_channels, int kernel_size, int stride, int padding, Device device = Device::CPU); 
    ~Convolution() {}
    Tensor_ptr forward(Tensor_ptr input) override;
    std::vector<Tensor_ptr> params() { return {weights, biases}; }
    Tensor_ptr get_weights() { return weights; }
    Tensor_ptr get_biases() { return biases; }

};

class Softmax : public Component {
public:
    Softmax() {};
    Tensor_ptr forward(Tensor_ptr input) override;
};
