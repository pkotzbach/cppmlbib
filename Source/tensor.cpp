#include "tensor.hpp"
#include "cpu_ops.hpp"
#include "cuda_ops.hpp"

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
    strides.resize(shape.size());
    strides.back() = 1;

    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // init values
    if (init_values.size() > 0 && init_values.size() != total_count) throw std::runtime_error("init values doesnt match shape");
    if (init_grads.size() > 0 && init_grads.size() != total_count) throw std::runtime_error("init grads doesnt match shape");

    values = std::shared_ptr<double[]>(new double[total_count], std::default_delete<double[]>());
    grads  = std::shared_ptr<double[]>(new double[total_count], std::default_delete<double[]>());

    for (int i = 0; i < total_count; ++i)
    {
        if (init_values.size() > 0) values[i] = init_values[i];
        else values[i] = init_zero? 0: sample_kaiming(total_count);
    }

    for (int i = 0; i < total_count; ++i)
    {
        if (init_grads.size() > 0) grads[i] = init_grads[i];
        else grads[i] = 0;
    }

    return *this;
}

// TODO: dont copy code

// returns continous values
std::vector<double> Tensor::values_vec()
{
    std::vector<double> vec;
    vec.resize(total_count);
    for (int i = 0; i < total_count; ++i) vec[i] = at(i);
    return vec;
}

std::vector<double> Tensor::grads_vec()
{
    std::vector<double> vec;
    vec.resize(total_count);
    for (int i = 0; i < total_count; ++i) vec[i] = grad_at(i);
    return vec;
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

int Tensor::strided_idx(std::vector<int> indices)
{
    if (indices.size() != shape.size()) throw std::runtime_error("wrong indices vector");

    int strided_idx = 0;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape[i]) throw std::runtime_error("wrong indices vector");
        strided_idx += indices[i] * strides[i];
    }

    return strided_idx;
}

int Tensor::strided_idx(int shape_idx, std::vector<int>& strides, std::vector<int>& shape)
{
    if (strides == shape) return shape_idx;
    
    int strided_idx = 0, temp;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        temp = shape_idx % shape[i];
        shape_idx = shape_idx / shape[i];
        strided_idx += temp * strides[i];
    }
    return strided_idx;
}

Tensor_ptr Tensor::relu()
{
    auto result = Tensor::init(shape, true);
    result->parents = std::pair{shared_from_this(), nullptr};
    
    for (int i = 0; i < total_count; ++i) {
        result->at(i) = at(i) > 0? at(i): 0;
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grad_at(i) += (r->at(i) > 0 ? 1 : 0) * r->grad_at(i);
            }
        }
    };

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

std::vector<int> broadcast_shape(std::vector<int> &a, std::vector<int> &b)
{
    int ndim = std::max(a.size(), b.size());
    std::vector<int> result(ndim);

    for (int i = 0; i < ndim; ++i) {
            int ai = i < a.size() ? a[a.size() - 1 - i] : 1;
            int bi = i < b.size() ? b[b.size() - 1 - i] : 1;

            if (ai != bi && ai != 1 && bi != 1) throw std::invalid_argument("Incompatible broadcast shapes");

            result[ndim - 1 - i] = std::max(ai, bi);
        }

    return result;
}

std::vector<int> Tensor::broadcast_strides(int ndim)
{
    std::vector<int> result(ndim, 0);
    int offset = ndim - shape.size();

    for (int i = 0; i < shape.size(); ++i)
    {
        if (shape[i] != 1) result[offset + i] = strides[i];
    }

    return result;
}

Tensor_ptr Tensor::sum()
{
    auto result = Tensor::init({1}, true);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "sum";

    for (int i = 0; i < total_count; ++i) {
        result->values[0] += at(i);
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->parents.first->total_count; ++i) {
                r->parents.first->grad_at(i) += r->grads[0];
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


Tensor_ptr Tensor::exp()
{
    auto result = Tensor::init(shape, true);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "exp";

    // TODO: at not needed here i think, could use buffer explicitly
    for (int i = 0; i < total_count; ++i) {
        result->at(i) = std::exp(at(i));
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grad_at(i) += r->grad_at(i) * r->at(i);
            }
        }
    };

    return result;
}

//tensor operators
Tensor_ptr operator+(Tensor_ptr self, Tensor_ptr other)
{
    if (self->get_device() != other->get_device()) throw std::invalid_argument("Tensors must be on the same device");
    std::string device = self->device;
    std::vector<int> out_shape = broadcast_shape(self->shape, other->shape);
    int ndim = out_shape.size();
    auto self_strides = self->broadcast_strides(ndim);
    auto other_strides = other->broadcast_strides(ndim);

    auto result = Tensor::init(out_shape, true, device);
    result->parents = std::pair{self, other};
    result->op = "add";

    if (device == "cpu") {
        for (int i = 0; i < result->total_count; ++i) {
            result->at(i) += self->at(i, self_strides, out_shape) + other->at(i, other_strides, out_shape);
        }
    }
    else if (device == "cuda") {
        std::vector<double> op_result = cuda::simple_op('+', self->values_vec(), other->values_vec(), self->get_total_count());
        for (int i = 0; i < result->total_count; ++i)
            result->values[i] = op_result[i];
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), first_str = self_strides, second_str = other_strides, out_shape](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grad_at(i, first_str, out_shape) += r->grad_at(i);
                r->parents.second->grad_at(i, second_str, out_shape) += r->grad_at(i);
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
        result->at(i) = self->at(i) - other->at(i);
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grad_at(i) += r->grad_at(i);
                r->parents.second->grad_at(i) -= r->grad_at(i);
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
        result->at(i) = self->at(i) * other->at(i);
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grad_at(i) += r->grad_at(i) * r->parents.second->at(i);
                r->parents.second->grad_at(i) += r->grad_at(i) * r->parents.first->at(i);
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
        result->at(i) = self->at(i) / other->at(scalar_div? 0: i);
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result), scalar_div](){
        if(auto r = res.lock()){
            for (int i = 0; i < r->total_count; ++i) {
                r->parents.first->grad_at(i) += r->grad_at(i) * (1.0f / r->parents.second->at(scalar_div? 0: i));
                r->parents.second->grad_at(scalar_div? 0: i) += r->grad_at(i) * 
                    -(r->parents.first->at(i) / (r->parents.second->at(scalar_div? 0: i) * r->parents.second->at(scalar_div? 0: i)));
            }
        }
    };
    return result;
}

void check_errors(Tensor_ptr t1, Tensor_ptr t2)
{
}

// Tensor_ptr Tensor::operator=(Tensor_ptr tensor)
// {
//     values = tensor->values;
//     shape = tensor->shape;
//     total_count = tensor->total_count;
//     device = tensor->device;
//     return shared_from_this();
// }

Tensor_ptr Tensor::matmul(Tensor_ptr tensor)
{
    if (shape.size() != 2 || tensor->shape.size() != 2) throw std::invalid_argument("Matmul defined only for 2d tensors");
    if (shape[1] != tensor->shape[0]) throw std::invalid_argument("Invalid shapes for matmul");
    if (device != tensor->device) throw std::invalid_argument("different devices");

    Tensor_ptr result = Tensor::init({shape[0], tensor->shape[1]}, true, device);
    result->parents = std::pair{shared_from_this(), tensor};
    result->op = "matmul";
    
    std::vector<double> matmul_output;
    if (device == "cpu") {
        matmul_output = cpu::matmul(values_vec(), tensor->values_vec(), shape[1], tensor->shape[1], shape[0]);
    } else if (device == "cuda") {
        matmul_output = cuda::matmul(values_vec(), tensor->values_vec(), shape[1], tensor->shape[1], shape[0]);
    }

    for (int i = 0; i < matmul_output.size(); ++i)
        result->values[i] = matmul_output[i];
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            Tensor_ptr firstT = r->parents.first->transpose();
            Tensor_ptr secondT = r->parents.second->transpose();
            std::vector<double> grad_first;
            std::vector<double> grad_second;
            if (r->get_device() == "cpu") {
                grad_first = cpu::matmul(r->grads_vec(), secondT->values_vec(), r->shape[1], secondT->shape[1], r->shape[0]);
                grad_second = cpu::matmul(firstT->values_vec(), r->grads_vec(), firstT->shape[1], r->shape[1], firstT->shape[0]);
            } else if (r->get_device() == "cuda") {
                grad_first = cuda::matmul(r->grads_vec(), secondT->values_vec(), r->shape[1], secondT->shape[1], r->shape[0]);
                grad_second = cuda::matmul(firstT->values_vec(), r->grads_vec(), firstT->shape[1], r->shape[1], firstT->shape[0]);
            }
            for (int i = 0; i < r->parents.first->total_count; ++i) {
                r->parents.first->grad_at(i) += grad_first[i];
            }
            for (int i = 0; i < r->parents.second->total_count; ++i) {
                r->parents.second->grad_at(i) += grad_second[i];
            }
        }
    };

    return result;
}

Tensor_ptr Tensor::softmax()
{
    if (shape.size() != 2) throw std::invalid_argument("Softmax defined only for 2d tensors");
    
    int N = shape[0];
    int C = shape[1];

    auto result = Tensor::init(shape, true);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "softmax";

    if (device == "cpu") {
        std::vector<double> exps(total_count);
        std::vector<double> sum_exps(N);

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                exps[n * C + c] = std::exp(at({n, c}));
                sum_exps[n] += exps[n * C + c];
            }
        }

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                result->at({n, c}) = exps[n * C + c] / sum_exps[n];
            }
        }
    } else if (device == "cuda") {
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), N, C](){
        if(auto r = res.lock()){
            for (int n = 0; n < N; ++n) {
                double sum = 0.0;
                for (int c = 0; c < C; ++c) {
                    sum += r->grad_at({n, c}) * r->at({n, c});
                }
                for (int c1 = 0; c1 < C; ++c1) {
                    r->parents.first->grad_at({n, c1}) += r->at({n, c1}) * (r->grad_at({n, c1}) - sum);
                }
            }
        }
    };

    return result;
}

Tensor_ptr Tensor::transpose()
{
    // TODO: i dont like that
    Tensor_ptr result = std::make_shared<Tensor>();
    result->values = values;
    result->grads = grads; // shouldnt it be separate buffer?
    result->total_count = total_count;
    result->shape = shape;
    result->strides = strides;
    result->device = device;
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "transpose";
    std::reverse(result->shape.begin(), result->shape.end());
    std::reverse(result->strides.begin(), result->strides.end());
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
            // printf("grad %s before %f\n", t->op.c_str(), t->grads[0]);
            t->backward_fn();
            // printf("grad %s after %f\n", t->op.c_str(), t->grads[0]);
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
    for (int i = 0; i < total_count; ++i) grad_at(i) = 0;
    if (parents.first && parents.first.get() != this)   parents.first->zero_grad();
    if (parents.second && parents.second.get() != this) parents.second->zero_grad();
}