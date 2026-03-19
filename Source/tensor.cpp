#include "tensor.hpp"
#include "cpu_ops.hpp"
#include "cuda_ops.hpp"

#include <stdexcept>
#include <cassert>
#include <random>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

float sample_kaiming(float n) {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0, std::sqrt(2/n));
    return dist(gen);
}

Storage::Storage(std::string i_device, std::vector<float> i_values, int i_size) : device(i_device), size(i_size)
{
    int memory = size * sizeof(float);
    if (device == "cpu") {
        data = std::shared_ptr<float[]>(new float[size], std::default_delete<float[]>());
        std::memcpy(data.get(), i_values.data(), memory);
    }
    else {
        float* raw_values = nullptr;
        cudaMalloc(&raw_values, memory);
        cudaMemcpy(raw_values, i_values.data(), memory, cudaMemcpyHostToDevice);
        data = std::shared_ptr<float[]>(raw_values, [](float* ptr) {cudaFree(ptr);});
    }
}

std::vector<float> Storage::cpu()
{
    if (device == "cuda") {
        std::vector<float> result(size);
        cudaMemcpy(result.data(), data.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
    return {};
}

void Storage::set(int idx, float val) {
    if (device == "cpu") {
        data[idx] = val;
    } else if (device == "cuda") {
        cudaMemcpy(data.get() + idx, &val, sizeof(float), cudaMemcpyHostToDevice);
    }
}

float Storage::at(int idx) {
    if (device == "cpu") {
        return data[idx];
    } else if (device == "cuda") {
        float val;
        cudaMemcpy(&val, data.get() + idx, sizeof(float), cudaMemcpyDeviceToHost);
        return val;
    }
    return 0;
}

float& Storage::operator[](int idx)
{
    if (device != "cpu") {
        throw std::runtime_error("operator[] not supported for CUDA storage");
    }
    return data[idx];
}

Tensor& Tensor::init_internal(std::vector<int> shape, std::vector<float> init_values, std::vector<float> init_grads, bool init_zero, std::string device) {
    if (shape.size() == 0) throw std::runtime_error("shape 0");
    if (device.compare("cuda") != 0 && device.compare("cpu") != 0) throw std::runtime_error("invalid device!");
    this->shape = shape;
    this->device = device;

    // calc total_count TODO: remove?
    total_count = shape[0];
    for (size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] <= 0) throw std::runtime_error("shape <= 0");
        total_count *= shape[i];
    }

    // calc strides
    strides = stride::calc_strides(shape);

    // init values
    if (init_values.size() > 0 && init_values.size() != total_count) throw std::runtime_error("init values doesnt match shape");
    if (init_grads.size() > 0 && init_grads.size() != total_count) throw std::runtime_error("init grads doesnt match shape");

    if (init_values.size() == 0) { 
        init_values.resize(total_count);
        if (init_zero) {
            std::fill(init_values.begin(), init_values.end(), 0);
        } else {
            for (auto& v : init_values) {
                v = sample_kaiming(total_count);
            }
        }
    }
    if (init_grads.size() == 0) {
        init_grads.assign(total_count, 0);
    }    

    values = Storage(device, init_values, total_count);
    grads = Storage(device, init_grads, total_count);

    return *this;
}

// returns continous values
std::vector<float> Tensor::values_vec(int count, std::vector<int>& strides, std::vector<int>& shape)
{
    std::vector<float> vec;
    vec.resize(count);
    for (int i = 0; i < count; ++i) vec[i] = get(i, strides, shape);
    return vec;
}

std::vector<float> Tensor::grads_vec() {
    std::vector<float> vec;
    vec.resize(total_count);
    for (int i = 0; i < total_count; ++i) vec[i] = grad_get(i);
    return vec;
}


Tensor_ptr Tensor::init(std::vector<int> shape, bool init_zero, std::string device) {
    auto t = std::make_shared<Tensor>();
    t->init_internal(shape, {}, {}, init_zero, device);
    return t;
}

Tensor_ptr Tensor::init(std::vector<int> shape, std::vector<float> values, std::string device) {
    auto t = std::make_shared<Tensor>();
    t->init_internal(shape, values, {}, false, device);
    return t;
}

// Tensor_ptr Tensor::init(std::vector<int> shape, std::vector<float> values, std::string device = "cpu")
// {
//     auto t = Tensor::init();
//     t->init_internal(shape, values, {}, false, device);
//     return t;
// }

Tensor_ptr Tensor::relu() {
    auto result = Tensor::init(shape, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};
    
    if (device == "cpu") {
        for (int i = 0; i < total_count; ++i) {
            result->set(i, get(i) > 0? get(i): 0);
        }
    } else if (device == "cuda") {
        cuda::relu(values.get(), result->values.get(), total_count);
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, r->parents.first->grad_get(i) + (r->get(i) > 0 ? 1 : 0) * r->grad_get(i));
                }
            } else if (r->device == "cuda") {
                cuda::relu_backward(r->parents.first->values.get(), r->parents.first->grads.get(), r->grads.get(), r->total_count);
            }
        }
    };

    return result;
}

Tensor_ptr Tensor::argmax(int axis) {
    // assume 2d matrix
    if (axis != 1) throw std::invalid_argument("Only for dim 1 now");
    if (shape.size() != 2) throw std::invalid_argument("Only for 2d matrix now");

    std::vector<int> result_shape;
    for (int i = 0; i < shape.size(); ++i) {
        result_shape.push_back(i == axis ? 1 : shape[i]);
    }
    Tensor_ptr result = Tensor::init(result_shape, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};

    int idx = -1;
    float max_val, data_val;
    for (int i = 0; i < shape[0]; ++i) {
        max_val = -INFINITY;
        idx = -1;
        for (int j = 0; j < shape[axis]; ++j) {
            data_val = get({i, j});
            if (data_val > max_val) {
                max_val = data_val;
                idx = j;
            }
        }
        result->set({i, 0}, (float)idx);
    }
    
    return result;
}

Tensor_ptr Tensor::max() {
    Tensor_ptr result = Tensor::init({1}, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};
    float max_val = -INFINITY;

    if (device == "cpu") {
        for (int i = 0; i < total_count; ++i) {
            if (values.at(i) > max_val) max_val = values.at(i);
        }
    } else if (device == "cuda") {
        max_val = cuda::reduction(ReductionOp::MAX, std::span<float>(values.get(), total_count));
    }

    result->values.set(0, max_val);

    return result;
}


Tensor_ptr Tensor::sum() {
    auto result = Tensor::init({1}, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "sum";

    if (device == "cpu") {
        for (int i = 0; i < total_count; ++i) {
            result->values.set(0, result->values.at(0) + get(i));
        }
    } else if (device == "cuda") {
        result->values.set(0, cuda::reduction(ReductionOp::SUM, std::span<float>(values.get(), total_count)));
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                for (int i = 0; i < r->parents.first->total_count; ++i) {
                    r->parents.first->grad_set(i, r->parents.first->grad_get(i) + r->grads.at(0));
                }
            } else if (r->device == "cuda") {
                cuda::sum_backward(r->parents.first->grads.get(), r->grads.get(), r->parents.first->total_count);
            }
        }
    };

    return result;
}

Tensor_ptr Tensor::sum(int axis) {
    if (shape.size() != 2) throw std::invalid_argument("sum(axis) only implemented for 2D");
    int N = shape[0];
    int C = shape[1];
    auto result = Tensor::init({axis == 0? C: N}, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "sum_ax" + std::to_string(axis);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            result->values.set(axis == 0? c: n, result->values.at(axis == 0? c: n) + values.at(n * C + c));
        }
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), N, C, axis](){
        if (auto r = res.lock()){
            if (r->device == "cpu") {
                for (int n = 0; n < N; ++n) {
                    for (int c = 0; c < C; ++c) {
                        r->parents.first->grad_set(n * C + c, r->parents.first->grad_get(n * C + c) + r->grad_get(axis == 0? c: n));
                    }
                }
            } else if (r->device == "cuda") {
                cuda::sum_axis_backward(r->parents.first->grads.get(), r->grads.get(), N, C, axis);
            }
        }
    };
    return result;
}


Tensor_ptr Tensor::exp() {
    auto result = Tensor::init(shape, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "exp";

    if (device == "cpu") {
        for (int i = 0; i < total_count; ++i) {
            result->set(i, std::exp(get(i)));
        }
    } else if (device == "cuda") {
        cuda::exp(values.get(), result->values.get(), total_count);
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, r->parents.first->grad_get(i) + r->grad_get(i) * r->get(i));
                }
            } else if (r->device == "cuda") {
                cuda::exp_backward(r->values.get(), r->parents.first->grads.get(), r->grads.get(), r->total_count);
            }
        }
    };

    return result;
}

// binary tensor operators

// TODO: should this struct be here?
struct BinaryOpContext {
    std::string device;
    std::vector<int> out_shape;
    std::vector<int> first_strides;
    std::vector<int> second_strides;

    BinaryOpContext(Tensor_ptr first, Tensor_ptr second)
    {
        if (first->get_device() != second->get_device()) {
            throw std::invalid_argument("Tensors must be on the same device");
        }
        
        device = first->device;
        out_shape = stride::broadcast_shape(first->shape, second->shape);
        int ndim = out_shape.size();
        
        first_strides = stride::broadcast_strides(first->shape, first->strides, ndim);
        second_strides = stride::broadcast_strides(second->shape, second->strides, ndim);
    }

    template <typename CPU_Op>
    void compute_forward(Tensor_ptr first, Tensor_ptr second, Tensor_ptr result, char cuda_op, CPU_Op cpu_op)
    {
        if (device == "cpu") {
            for (int i = 0; i < result->total_count; ++i) {
                result->set(i, cpu_op(first->get(i, first_strides, out_shape), second->get(i, second_strides, out_shape)));
            }
        }
        else if (device == "cuda") {
            int size = result->total_count;
            int dims = out_shape.size();
            // TODO: it looks bad
            std::array<int, MAX_DIMS> shape_arr;
            std::copy(out_shape.begin(), out_shape.end(), shape_arr.begin());

            std::array<int, MAX_DIMS> first_strides_arr;
            std::copy(first_strides.begin(), first_strides.end(), first_strides_arr.begin());

            std::array<int, MAX_DIMS> second_strides_arr;
            std::copy(second_strides.begin(), second_strides.end(), second_strides_arr.begin());
            cuda::binary_op_strided(cuda_op, first->raw_values(), first_strides_arr, second->raw_values(), second_strides_arr,
                                    shape_arr, size, dims, result->raw_values());
        }
    }
};

Tensor_ptr operator+(Tensor_ptr first, Tensor_ptr second) {
    BinaryOpContext ctx(first, second);

    auto result = Tensor::init(ctx.out_shape, true, ctx.device);
    result->parents = std::pair{first, second};
    result->op = "add";

    ctx.compute_forward(first, second, result, '+', [](float a, float b){ return a + b; });

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), ctx](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, ctx.first_strides, ctx.out_shape, r->parents.first->grad_get(i, ctx.first_strides, ctx.out_shape) + r->grad_get(i));
                    r->parents.second->grad_set(i, ctx.second_strides, ctx.out_shape, r->parents.second->grad_get(i, ctx.second_strides, ctx.out_shape) + r->grad_get(i));
                }
            } else if (r->device == "cuda") {
                int dims = ctx.out_shape.size();
                std::array<int, MAX_DIMS> shape_arr;
                std::copy(ctx.out_shape.begin(), ctx.out_shape.end(), shape_arr.begin());

                std::array<int, MAX_DIMS> first_strides_arr;
                std::copy(ctx.first_strides.begin(), ctx.first_strides.end(), first_strides_arr.begin());

                std::array<int, MAX_DIMS> second_strides_arr;
                std::copy(ctx.second_strides.begin(), ctx.second_strides.end(), second_strides_arr.begin());

                cuda::binary_op_backward_strided('+', r->parents.first->raw_values(), first_strides_arr,
                                               r->parents.second->raw_values(), second_strides_arr,
                                               r->parents.first->raw_grads(), r->parents.second->raw_grads(),
                                               r->raw_grads(), shape_arr, r->total_count, dims);
            }
        }
    };
    
    return result;
}

Tensor_ptr operator-(Tensor_ptr first, Tensor_ptr second) {
    BinaryOpContext ctx(first, second);

    auto result = Tensor::init(ctx.out_shape, true, ctx.device);
    result->parents = std::pair{first, second};
    result->op = "sub";

    ctx.compute_forward(first, second, result, '-', [](float a, float b){ return a - b; });
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result), ctx](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, ctx.first_strides, ctx.out_shape, r->parents.first->grad_get(i, ctx.first_strides, ctx.out_shape) + r->grad_get(i));
                    r->parents.second->grad_set(i, ctx.second_strides, ctx.out_shape, r->parents.second->grad_get(i, ctx.second_strides, ctx.out_shape) - r->grad_get(i));
                }
            } else if (r->device == "cuda") {
                int dims = ctx.out_shape.size();
                std::array<int, MAX_DIMS> shape_arr;
                std::copy(ctx.out_shape.begin(), ctx.out_shape.end(), shape_arr.begin());

                std::array<int, MAX_DIMS> first_strides_arr;
                std::copy(ctx.first_strides.begin(), ctx.first_strides.end(), first_strides_arr.begin());

                std::array<int, MAX_DIMS> second_strides_arr;
                std::copy(ctx.second_strides.begin(), ctx.second_strides.end(), second_strides_arr.begin());

                cuda::binary_op_backward_strided('-', r->parents.first->raw_values(), first_strides_arr,
                                               r->parents.second->raw_values(), second_strides_arr,
                                               r->parents.first->raw_grads(), r->parents.second->raw_grads(),
                                               r->raw_grads(), shape_arr, r->total_count, dims);
            }
        }
    };
    return result;
}

Tensor_ptr operator*(Tensor_ptr first, Tensor_ptr second) {
    BinaryOpContext ctx(first, second);

    auto result = Tensor::init(ctx.out_shape, true, ctx.device);
    result->parents = std::pair{first, second};
    result->op = "mul";

    ctx.compute_forward(first, second, result, '*', [](float a, float b){ return a * b; });
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result), ctx](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, ctx.first_strides, ctx.out_shape, r->parents.first->grad_get(i, ctx.first_strides, ctx.out_shape) + r->grad_get(i) * r->parents.second->get(i, ctx.second_strides, ctx.out_shape));
                    r->parents.second->grad_set(i, ctx.second_strides, ctx.out_shape, r->parents.second->grad_get(i, ctx.second_strides, ctx.out_shape) + r->grad_get(i) * r->parents.first->get(i, ctx.first_strides, ctx.out_shape));
                }
            } else if (r->device == "cuda") {
                int dims = ctx.out_shape.size();
                std::array<int, MAX_DIMS> shape_arr;
                std::copy(ctx.out_shape.begin(), ctx.out_shape.end(), shape_arr.begin());

                std::array<int, MAX_DIMS> first_strides_arr;
                std::copy(ctx.first_strides.begin(), ctx.first_strides.end(), first_strides_arr.begin());

                std::array<int, MAX_DIMS> second_strides_arr;
                std::copy(ctx.second_strides.begin(), ctx.second_strides.end(), second_strides_arr.begin());

                cuda::binary_op_backward_strided('*', r->parents.first->raw_values(), first_strides_arr,
                                               r->parents.second->raw_values(), second_strides_arr,
                                               r->parents.first->raw_grads(), r->parents.second->raw_grads(),
                                               r->raw_grads(), shape_arr, r->total_count, dims);
            }
        }
    };
    return result;
}

Tensor_ptr operator/(Tensor_ptr first, Tensor_ptr second) {
    BinaryOpContext ctx(first, second);

    auto result = Tensor::init(ctx.out_shape, true, ctx.device);
    result->parents = std::pair{first, second};
    result->op = "div";

    ctx.compute_forward(first, second, result, '/', [](float a, float b){ return a / b; });

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), ctx](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, ctx.first_strides, ctx.out_shape, r->parents.first->grad_get(i, ctx.first_strides, ctx.out_shape) + r->grad_get(i) * (1.0f / r->parents.second->get(i, ctx.second_strides, ctx.out_shape)));
                    r->parents.second->grad_set(i, ctx.second_strides, ctx.out_shape, r->parents.second->grad_get(i, ctx.second_strides, ctx.out_shape) + r->grad_get(i) * 
                        -(r->parents.first->get(i, ctx.first_strides, ctx.out_shape) / (r->parents.second->get(i, ctx.second_strides, ctx.out_shape) * r->parents.second->get(i, ctx.second_strides, ctx.out_shape))));
                }
            } else if (r->device == "cuda") {
                int dims = ctx.out_shape.size();
                std::array<int, MAX_DIMS> shape_arr;
                std::copy(ctx.out_shape.begin(), ctx.out_shape.end(), shape_arr.begin());

                std::array<int, MAX_DIMS> first_strides_arr;
                std::copy(ctx.first_strides.begin(), ctx.first_strides.end(), first_strides_arr.begin());

                std::array<int, MAX_DIMS> second_strides_arr;
                std::copy(ctx.second_strides.begin(), ctx.second_strides.end(), second_strides_arr.begin());

                cuda::binary_op_backward_strided('/', r->parents.first->raw_values(), first_strides_arr,
                                               r->parents.second->raw_values(), second_strides_arr,
                                               r->parents.first->raw_grads(), r->parents.second->raw_grads(),
                                               r->raw_grads(), shape_arr, r->total_count, dims);
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

Tensor_ptr Tensor::matmul(Tensor_ptr tensor) {
    if (shape.size() != 2 || tensor->shape.size() != 2) throw std::invalid_argument("Matmul defined only for 2d tensors");
    if (shape[1] != tensor->shape[0]) throw std::invalid_argument("Invalid shapes for matmul");
    if (device != tensor->device) throw std::invalid_argument("different devices");

    Tensor_ptr result = Tensor::init({shape[0], tensor->shape[1]}, true, device);
    result->parents = std::pair{shared_from_this(), tensor};
    result->op = "matmul";
    
    std::vector<float> matmul_output;
    if (device == "cpu") {
        matmul_output = cpu::matmul(values_vec(), tensor->values_vec(), shape[1], tensor->shape[1], shape[0]);
    } else if (device == "cuda") {
        matmul_output = cuda::matmul(values_vec(), tensor->values_vec(), shape[1], tensor->shape[1], shape[0]);
    }

    for (int i = 0; i < matmul_output.size(); ++i)
        result->values.set(i, matmul_output[i]);
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                Tensor_ptr firstT = r->parents.first->transpose();
                Tensor_ptr secondT = r->parents.second->transpose();
                std::vector<float> grad_first;
                std::vector<float> grad_second;
                grad_first = cpu::matmul(r->grads_vec(), secondT->values_vec(), r->shape[1], secondT->shape[1], r->shape[0]);
                grad_second = cpu::matmul(firstT->values_vec(), r->grads_vec(), firstT->shape[1], r->shape[1], firstT->shape[0]);
                for (int i = 0; i < r->parents.first->total_count; ++i) {
                    r->parents.first->grad_set(i, r->parents.first->grad_get(i) + grad_first[i]);
                }
                for (int i = 0; i < r->parents.second->total_count; ++i) {
                    r->parents.second->grad_set(i, r->parents.second->grad_get(i) + grad_second[i]);
                }
            } else if (r->device == "cuda") {
                cuda::matmul_backward(r->parents.first->values.get(), r->parents.second->values.get(), 
                                      r->parents.first->grads.get(), r->parents.second->grads.get(), 
                                      r->grads.get(), r->parents.first->shape[1], r->parents.second->shape[1], r->parents.first->shape[0]);
            }
        }
    };

    return result;
}

Tensor_ptr Tensor::softmax() {
    if (shape.size() != 2) throw std::invalid_argument("Softmax defined only for 2d tensors");
    
    int N = shape[0];
    int C = shape[1];

    auto result = Tensor::init(shape, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "softmax";

    if (device == "cpu") {
        cpu::softmax(values_vec(), result->values.get(), N, C);
    } else if (device == "cuda") {
        cuda::make_continous(shared_from_this());
        cuda::softmax(values.get(), result->values.get(), N, C);
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), N, C](){
        if(auto r = res.lock()){
            if (r->device == "cpu") {
                for (int n = 0; n < N; ++n) {
                    float sum = 0.0;
                    for (int c = 0; c < C; ++c) {
                        sum += r->grad_get({n, c}) * r->get({n, c});
                    }
                    for (int c1 = 0; c1 < C; ++c1) {
                        r->parents.first->grad_set({n, c1}, r->parents.first->grad_get({n, c1}) + r->get({n, c1}) * (r->grad_get({n, c1}) - sum));
                    }
                }
            } else if (r->device == "cuda") {
                cuda::softmax_backward(r->values.get(), r->parents.first->grads.get(), r->grads.get(), N, C);
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

void Tensor::backward() {
    if (total_count != 1) throw std::invalid_argument("backward only possible on 1x1 tensor");
    std::unordered_set<Tensor_ptr> visited{};
    std::vector<Tensor_ptr> topo{};
    toposort(shared_from_this(), visited, topo);
    std::reverse(topo.begin(), topo.end());
    grads.set(0, 1.0f);

    for (auto t : topo) {
        if (t->backward_fn) {
            t->backward_fn();
        }
    }
}

void Tensor::toposort(Tensor_ptr t, std::unordered_set<Tensor_ptr>& visited, std::vector<Tensor_ptr>& res) {
    if (visited.find(t) == visited.end()) {
        visited.insert(t);
        if (t->parents.first)
            toposort(t->parents.first, visited, res);
        if (t->parents.second)
            toposort(t->parents.second, visited, res);
        res.push_back(t);
    }
}

void Tensor::zero_grad() {
    if (device == "cpu") {
        for (int i = 0; i < total_count; ++i) grad_set(i, 0);
    } else if (device == "cuda") {
        cudaMemset(grads.get(), 0, total_count * sizeof(float));
    }
    if (parents.first && parents.first.get() != this)   parents.first->zero_grad();
    if (parents.second && parents.second.get() != this) parents.second->zero_grad();
}
