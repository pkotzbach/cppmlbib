#include "tensor.hpp"

#include <stdexcept>
#include <cassert>
#include <random>
#include <string>

double sample_kaiming(double n)
{
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, std::sqrt(2/n));
    return dist(gen);
}

Tensor &Tensor::init_internal(std::vector<int> shape, std::vector<double> init_values, std::vector<double> init_grads, bool init_zero, std::string device)
{
    if (shape.size() == 0) throw std::runtime_error("shape 0");
    if (device.compare("cuda") != 0 && device.compare("cpu") != 0) throw std::runtime_error("invalid device!");
    this->shape = shape;
    this->device = device;

    // calc total_count TODO: remove?
    total_count = shape[0];
    for (size_t i = 1; i < shape.size(); ++i)
    {
        if (shape[i] <= 0) throw std::runtime_error("shape <= 0");
        total_count *= shape[i];
    }

    // calc strides
    // strides.push_back(1);
    // for (int i = 1; i < shape.size(); ++i) {
    //     strides.push_back(strides[i-1] * shape[i]);
    // }
    // std::reverse(strides.begin(), strides.end());

    // init values
    if (init_values.size() > 0 && init_values.size() != total_count) throw std::runtime_error("init values doesnt match shape");
    if (init_grads.size() > 0 && init_grads.size() != total_count) throw std::runtime_error("init grads doesnt match shape");

    values = new double[total_count];
    for (int i = 0; i < total_count; ++i)
    {
        if (init_values.size() > 0) values[i] = init_values[i];
        else values[i] = init_zero? 0: sample_kaiming(total_count);
    }

    grads = new double[total_count];
    for (int i = 0; i < total_count; ++i)
    {
        if (init_grads.size() > 0) grads[i] = init_grads[i];
        else grads[i] = 0;
    }

    return *this;
}

Tensor::~Tensor()
{
    delete[] values;
    delete[] grads;
}

Tensor_ptr Tensor::init(std::vector<int> shape, bool init_zero, std::string device)
{
    auto t = std::make_shared<Tensor>();
    t->init_internal(shape, {}, {}, init_zero, device);
    return t;
}

Tensor_ptr Tensor::init(std::vector<int> shape, std::vector<double> values, std::string device)
{
    auto t = std::make_shared<Tensor>();
    t->init_internal(shape, values, {}, false, device);
    return t;
}

// Tensor_ptr Tensor::init(std::vector<int> shape, std::vector<double> values, std::string device = "cpu")
// {
//     auto t = Tensor::init();
//     t->init_internal(shape, values, {}, false, device);
//     return t;
// }

double& Tensor::_at(std::vector<int> indices, double* source)
{
    if (indices.size() < shape.size()) throw std::runtime_error("wrong indices vector");

    int idx = 0;
    for (int i = shape.size() - 1; i >= 0; --i) {
        idx += indices[i] * strides[i];
    }

    return source[idx];
}

double& Tensor::_at(int flat_idx, double* source)
{
    int strided_idx = 0, temp;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        temp = flat_idx % shape[i];
        flat_idx = flat_idx / shape[i];
        strided_idx += temp * strides[i];
    }
    return source[strided_idx];
}

Tensor_ptr Tensor::relu()
{
    auto result = Tensor::init(shape, true);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grads[i] += (r->values[i] > 0 ? 1 : 0) * r->grads[i];
            }
        }
    };

    for (int i = 0; i < total_count; ++i) {
        result->values[i] = values[i] > 0? values[i]: 0;
    }

    return result;
}

Tensor_ptr Tensor::argmax(int axis)
{
    // assume 2d matrix
    if (axis != 1) throw std::invalid_argument("Only for dim 1 now");
    if (shape.size() != 2) throw std::invalid_argument("Only for 2d matrix now");

    std::vector<int> result_shape;
    for (int i = 0; i < shape.size(); ++i) {
        result_shape.push_back(i == axis ? 1 : shape[i]);
    }
    Tensor_ptr result = Tensor::init(result_shape, true);

    int idx = -1;
    double max, data;
    for (int i = 0; i < shape[0]; ++i) {
        max = -INFINITY;
        idx = -1;
        for (int j = 0; j < shape[axis]; ++j) {
            data = at({i, j});
            if (data > max) {
                max = data;
                idx = j;
            }
        }
        result->at({i, 0}) = idx;
    }
    
    return result;
}

Tensor &Tensor::flatten()
{
    shape = {total_count};
    return *this;
}

Tensor_ptr Tensor::sum()
{
    auto result = Tensor::init({1}, true);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "sum";

    for (int i = 0; i < total_count; ++i) {
        result->values[0] += values[i];
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->parents.first->total_count; ++i) {
                r->parents.first->grads[i] += r->grads[0];
            }
        }
    };

    return result;
}

Tensor_ptr Tensor::sum(int axis)
{
    if (shape.size() != 2) throw std::invalid_argument("sum(axis) only implemented for 2D");
    int N = shape[0];
    int C = shape[1];
    auto result = Tensor::init({axis == 0? C: N}, true);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "sum_ax" + std::to_string(axis);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            result->values[axis == 0? c: n] += values[n * C + c];
        }
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), N, C, axis](){
        if (auto r = res.lock()){
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    r->parents.first->grads[n * C + c] += r->grads[axis == 0? c: n];
            }
        }
        }
    };
    return result;
}

// Tensor_ptr Tensor::max(int axis)
// {

// }

Tensor_ptr Tensor::exp()
{
    auto result = Tensor::init(shape, true);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "exp";

    for (int i = 0; i < total_count; ++i) {
        result->values[i] = std::exp(values[i]);
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grads[i] += r->grads[i] * r->values[i];
            }
        }
    };

    return result;
}

int Tensor::calc_strided_idx(int flat_idx)
{
    int strided_idx = 0, temp;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        temp = flat_idx % shape[i];
        flat_idx = flat_idx / shape[i];
        strided_idx += temp * strides[i];
    }
    return strided_idx;
}

//tensor operators
Tensor_ptr operator+(Tensor_ptr self, Tensor_ptr other)
{
    bool broadcast = (self->shape != other->shape) && self->shape[self->shape.size() - 1] == other->shape[0] && other->shape.size() == 1;
    if (!broadcast && self->shape != other->shape) throw std::invalid_argument("Shape must be the same or broadcast");
    auto result = Tensor::init(self->shape, true);
    result->parents = std::pair{self, other};
    result->op = "add";

    for (int i = 0; i < self->total_count; ++i) {

        result->values[i] += self->values[self->calc_strided_idx(i)] + other->values[other->calc_strided_idx(i)];
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grads[r->parents.first->calc_strided_idx(i)] += r->grads[i];
                r->parents.second->grads[i] += r->grads[i];
            }
        }
    };
    return result;
}

Tensor_ptr operator-(Tensor_ptr self, Tensor_ptr other)
{
    if (self->shape != other->shape) throw std::invalid_argument("Shape must be the same");
    auto result = Tensor::init(self->shape, true);
    result->parents = std::pair{self, other};
    result->op = "sub";

    for (int i = 0; i < self->total_count; ++i) {
        result->values[i] = self->values[i] - other->values[i];
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grads[i] += r->grads[i];
                r->parents.second->grads[i] -= r->grads[i];
            }
        }
    };
    return result;
}

Tensor_ptr operator*(Tensor_ptr self, Tensor_ptr other)
{
    if (self->shape != other->shape) throw std::invalid_argument("Shape must be the same");
    auto result = Tensor::init(self->shape, true);
    result->parents = std::pair{self, other};
    result->op = "mul";

    for (int i = 0; i < self->total_count; ++i) {
        result->values[i] = self->values[i] * other->values[i];
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grads[i] += r->grads[i] * r->parents.second->values[i];
                r->parents.second->grads[i] += r->grads[i] * r->parents.first->values[i];
            }
        }
    };
    return result;
}

Tensor_ptr operator/(Tensor_ptr self, Tensor_ptr other)
{
    bool scalar_div = (other->total_count == 1);
    if (self->shape != other->shape) throw std::invalid_argument("Shape must be the same or div by scalar");

    auto result = Tensor::init(self->shape, true);
    result->parents = std::pair{self, other};
    result->op = "div";

    for (int i = 0; i < self->total_count; ++i) {
        result->values[i] = self->values[i] / other->values[scalar_div? 0: i];
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result), scalar_div](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grads[i] += r->grads[i] * (1.0f / r->parents.second->values[scalar_div? 0: i]);
                r->parents.second->grads[scalar_div? 0: i] += r->grads[i] * 
                    -(r->parents.first->values[i] / (r->parents.second->values[scalar_div? 0: i] * r->parents.second->values[scalar_div? 0: i]));
            }
        }
    };
    return result;
}

// Tensor_ptr Tensor::operator=(Tensor_ptr tensor)
// {
//     values = tensor->values;
//     shape = tensor->shape;
//     total_count = tensor->total_count;
//     device = tensor->device;
//     return shared_from_this();
// }

std::vector<double> _matmul(double* A, double* B, int K, int X, int Y)
{
    std::vector<double> output;
    output.resize(X*Y);

    for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
            for (int k = 0; k < K; ++k) 
                output[y * X + x] = output[y * X + x] + A[y * K + k] * B[k * X + x];

    return output;
}

Tensor_ptr Tensor::matmul(Tensor_ptr tensor)
{
    if (shape.size() != 2 || tensor->shape.size() != 2) throw std::invalid_argument("Matmul defined only for 2d tensors");
    if (shape[1] != tensor->shape[0]) throw std::invalid_argument("Invalid shapes for matmul");

    Tensor_ptr result = Tensor::init({shape[0], tensor->shape[1]}, true);
    result->parents = std::pair{shared_from_this(), tensor};
    result->op = "matmul";
    
    // result->values = _matmul(values, tensor->values, shape[1], tensor->shape[1], shape[0]); //memory leak
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            Tensor_ptr firstT = r->parents.first->transpose();
            Tensor_ptr secondT = r->parents.second->transpose();
            std::vector<double> grad_first = _matmul(r->grads, secondT->values, r->shape[1], secondT->shape[1], r->shape[0]);
            std::vector<double> grad_second = _matmul(firstT->values, r->grads, firstT->shape[1], r->shape[1], firstT->shape[0]);
            for (int i = 0; i < r->parents.first->total_count; ++i) {
                r->parents.first->grads[i] += grad_first[i];
            }
            for (int i = 0; i < r->parents.second->total_count; ++i) {
                r->parents.second->grads[i] += grad_second[i];
            }
        }
    };

    return result;
}

Tensor_ptr Tensor::transpose()
{
    if (shape.size() != 2) throw std::invalid_argument("transpose defined only for 2d tensors");
    Tensor_ptr result = Tensor::init({shape[1], shape[0]}, false);
    for (int x = 0; x < shape[0]; ++x)
        for (int y = 0; y < shape[1]; ++y)
            result->at({y, x}) = at({x, y});
    return result;
}

void Tensor::backward()
{
    if (total_count != 1) throw std::invalid_argument("backward only possible on 1x1 tensor");
    std::unordered_set<Tensor_ptr> visited{};
    std::vector<Tensor_ptr> topo{};
    toposort(shared_from_this(), visited, topo); // weakptr?
    std::reverse(topo.begin(), topo.end());
    grads[0] = 1.0f;
    for (auto t : topo)
    {
        if (t->backward_fn)
        {
            printf("grad %c before %f\n", t->op, t->grads[0]);
            t->backward_fn();
            printf("grad %c after %f\n", t->op, t->grads[0]);
        }
    }
}

void Tensor::toposort(Tensor_ptr t, std::unordered_set<Tensor_ptr> &visited, std::vector<Tensor_ptr> &res)
{
    if (visited.find(t) == visited.end())
    {
        visited.insert(t);
        if (t->parents.first)
            toposort(t->parents.first, visited, res);
        if (t->parents.second)
            toposort(t->parents.second, visited, res);
        res.push_back(t);
    }
}

void Tensor::zero_grad()
{
    for (int i = 0; i < total_count; ++i) grads[i] = 0;
    if (parents.first && parents.first.get() != this)   parents.first->zero_grad();
    if (parents.second && parents.second.get() != this) parents.second->zero_grad();
}