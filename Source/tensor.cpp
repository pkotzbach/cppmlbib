#include "tensor.hpp"
#include "cpu_ops.hpp"
#include "cuda_ops.hpp"

#include <stdexcept>
#include <cassert>
#include <random>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <format>
#include <ranges>
#include <algorithm>

Storage::Storage(Device i_device, std::vector<float> i_values, int i_size) : size(i_size), device(i_device)
{
    int memory = size * sizeof(float);
    if (device == Device::CPU) {
        int alignment = 64; // AVX-512
        int memory_aligned = ((memory + alignment - 1) / alignment) * alignment;
        float* raw_ptr = static_cast<float*>(std::aligned_alloc(alignment, memory_aligned));
        if (!raw_ptr) throw std::bad_alloc();
        memcpy(raw_ptr, i_values.data(), memory);
        data = std::shared_ptr<float[]>(raw_ptr, [](float* ptr){ std::free(ptr); });
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
    std::vector<float> result(size);
    if (device == Device::CUDA) {
        cudaMemcpy(result.data(), data.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(result.data(), data.get(), size * sizeof(float));
    }
    return result;
}


Tensor& Tensor::init_internal(std::vector<int> shape, std::vector<float> init_values, std::vector<float> init_grads, bool init_zero, Device device) {
    if (shape.empty()) throw std::runtime_error("shape 0");
    this->shape = std::move(shape);
    this->device = device;

    // calc total_count TODO: remove?
    total_count = this->shape[0];
    for (size_t i = 1; i < this->shape.size(); ++i) {
        if (this->shape[i] <= 0) throw std::runtime_error("shape <= 0");
        total_count *= this->shape[i];
    }

    // calc strides
    strides = stride::calc_strides(this->shape);

    // init values
    if (!init_values.empty() && init_values.size() != static_cast<size_t>(total_count)) throw std::runtime_error("init values doesnt match shape");
    if (!init_grads.empty() && init_grads.size() != static_cast<size_t>(total_count)) throw std::runtime_error("init grads doesnt match shape");

    if (init_values.empty()) { 
        init_values.resize(total_count);
        if (init_zero) {
            std::ranges::fill(init_values, 0.0f);
        } else {
            // sample kaiming
            std::mt19937 gen(std::random_device{}());
            std::normal_distribution<float> dist(0.0, std::sqrt(2.0f/total_count));
            for (auto& v : init_values) {
                v = dist(gen);
            }
        }
    }
    if (init_grads.empty()) {
        init_grads.assign(total_count, 0);
    }    

    values = Storage(this->device, std::move(init_values), total_count);
    grads = Storage(this->device, std::move(init_grads), total_count);

    image = false;

    return *this;
}

bool Tensor::is_continous(const std::vector<int>& strides, const std::vector<int>& shape) {
    if (strides.empty() || shape.empty()) return true;
    return std::ranges::equal(strides, stride::calc_strides(shape));
}

// returns continous values
std::vector<float> Tensor::values_vec(int count, std::vector<int>& strides, std::vector<int>& shape)
{
    std::vector<float> vec(count);
    if (is_continous(strides, shape) && device == Device::CPU) {
        std::memcpy(vec.data(), raw_values(), count * sizeof(float));
        return vec;
    }
    else {
        for (int i = 0; i < count; ++i) vec[i] = get(i, strides, shape);
        return vec;
    }
}

std::vector<float> Tensor::grads_vec() {
    std::vector<float> vec(total_count);
    if (is_continous() && device == Device::CPU) {
        std::memcpy(vec.data(), raw_grads(), total_count * sizeof(float));
        return vec;
    }
    else {
        for (int i = 0; i < total_count; ++i) vec[i] = grad_get(i, strides, shape);
        return vec;
    }
}


Tensor_ptr Tensor::init(std::vector<int> shape, bool init_zero, Device device) {
    auto t = std::make_shared<Tensor>();
    t->init_internal(std::move(shape), {}, {}, init_zero, device);
    return t;
}

Tensor_ptr Tensor::init(std::vector<int> shape, std::vector<float> values, Device device) {
    auto t = std::make_shared<Tensor>();
    t->init_internal(std::move(shape), std::move(values), {}, false, device);
    return t;
}

void Tensor::print()
{
    std::vector<float> cpu_values = values.cpu();

    std::function<void(int, std::vector<int>&)> print_rec = [&](int dim, std::vector<int>& indices) {
        if (dim == (int)shape.size()) {
            std::cout << cpu_values[stride::strided_idx(indices, strides, shape)];
            return;
        }

        std::cout << "[";
        for (int i = 0; i < shape[dim]; ++i) {
            indices[dim] = i;
            print_rec(dim + 1, indices);
            if (i < shape[dim] - 1) {
                std::cout << (dim == (int)shape.size() - 1 ? ", " : ",\n" + std::string(dim + 1, ' '));
            }
        }
        std::cout << "]";
    };

    std::vector<int> indices(shape.size(), 0);
    print_rec(0, indices);
    std::cout << " device='" << (device == Device::CPU ? "cpu" : "cuda") << "', shape=(";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    std::cout << ")" << std::endl;
}

void Tensor::print_shape()
{
    std::cout << "[";
    for (int i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

Tensor_ptr Tensor::relu() {
    auto result = Tensor::init(shape, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};
    
    if (device == Device::CPU) {
        if (is_continous()) {
            float* result_raw = result->raw_values();
            float* input_raw = raw_values();
            for (int i = 0; i < total_count; ++i) {
                result_raw[i] = std::max(input_raw[i], 0.0f);
            }
        }
        else {
            for (int i = 0; i < total_count; ++i) {
            result->set(i, std::max(get(i), 0.0f));
            }
        }
    } else if (device == Device::CUDA) {
        cuda::relu(values.get(), result->values.get(), total_count);
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            if (r->device == Device::CPU) {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, r->parents.first->grad_get(i) + (r->get(i) > 0 ? 1 : 0) * r->grad_get(i));
                }
            } else if (r->device == Device::CUDA) {
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

    if (device == Device::CPU) {
        for (int i = 0; i < total_count; ++i) {
            if (values.at(i) > max_val) max_val = values.at(i);
        }
    } else if (device == Device::CUDA) {
        max_val = cuda::reduction(ReductionOp::MAX, std::span<float>(values.get(), total_count));
    }

    result->values.set(0, max_val);

    return result;
}


Tensor_ptr Tensor::sum() {
    auto result = Tensor::init({1}, true, device);
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "sum";

    if (device == Device::CPU) {
        for (int i = 0; i < total_count; ++i) {
            result->values.set(0, result->values.at(0) + get(i));
        }
    } else if (device == Device::CUDA) {
        result->values.set(0, cuda::reduction(ReductionOp::SUM, std::span<float>(values.get(), total_count)));
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            if (r->device == Device::CPU) {
                for (int i = 0; i < r->parents.first->total_count; ++i) {
                    r->parents.first->grad_set(i, r->parents.first->grad_get(i) + r->grads.at(0));
                }
            } else if (r->device == Device::CUDA) {
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
    result->op = std::format("sum_ax{}", axis);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            result->values.set(axis == 0? c: n, result->values.at(axis == 0? c: n) + values.at(n * C + c));
        }
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), N, C, axis](){
        if (auto r = res.lock()){
            if (r->device == Device::CPU) {
                for (int n = 0; n < N; ++n) {
                    for (int c = 0; c < C; ++c) {
                        r->parents.first->grad_set(n * C + c, r->parents.first->grad_get(n * C + c) + r->grad_get(axis == 0? c: n));
                    }
                }
            } else if (r->device == Device::CUDA) {
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

    if (device == Device::CPU) {
        for (int i = 0; i < total_count; ++i) {
            result->set(i, std::exp(get(i)));
        }
    } else if (device == Device::CUDA) {
        cuda::exp(values.get(), result->values.get(), total_count);
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            if (r->device == Device::CPU) {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, r->parents.first->grad_get(i) + r->grad_get(i) * r->get(i));
                }
            } else if (r->device == Device::CUDA) {
                cuda::exp_backward(r->values.get(), r->parents.first->grads.get(), r->grads.get(), r->total_count);
            }
        }
    };

    return result;
}

// binary tensor operators

// TODO: should this struct be here?
struct BinaryOpContext {
    Device device;
    std::vector<int> out_shape_vec;
    std::vector<int> first_strides_vec;
    std::vector<int> second_strides_vec;

    std::array<int, MAX_DIMS> out_shape{};
    std::array<int, MAX_DIMS> first_strides{};
    std::array<int, MAX_DIMS> second_strides{};
    int ndim;
    bool casted;

    BinaryOpContext(const Tensor_ptr& first, const Tensor_ptr& second)
    {
        if (first->get_device() != second->get_device()) {
            throw std::invalid_argument("Tensors must be on the same device");
        }
        
        device = first->get_device();
        out_shape_vec = stride::broadcast_shape(first->get_shape(), second->get_shape());
        ndim = out_shape_vec.size();
        
        first_strides_vec = stride::broadcast_strides(first->get_shape(), first->get_strides(), ndim);
        second_strides_vec = stride::broadcast_strides(second->get_shape(), second->get_strides(), ndim);

        casted = (first->get_shape() != out_shape_vec) || (second->get_shape() != out_shape_vec);

        std::ranges::copy(out_shape_vec, out_shape.begin());
        std::ranges::copy(first_strides_vec, first_strides.begin());
        std::ranges::copy(second_strides_vec, second_strides.begin());
    }
};

Tensor_ptr operator+(Tensor_ptr first, Tensor_ptr second) {
    BinaryOpContext ctx(first, second);

    auto result = Tensor::init(ctx.out_shape_vec, true, ctx.device);
    result->parents = std::pair{first, second};
    result->op = "add";

    if (ctx.device == Device::CPU) {
        float* result_raw = result->raw_values();
        bool handled = false;

        if (first->is_continous() && second->is_continous()) {
            float* first_raw = first->raw_values();
            float* second_raw = second->raw_values();

            if (!ctx.casted) {
                for (int i = 0; i < result->total_count; ++i) {
                    result_raw[i] = first_raw[i] + second_raw[i];
                }
                handled = true;
            }
            // specific target [N, C] + [1, C] for bias addition
            else if (ctx.ndim == 2 &&
                    ctx.first_strides_vec[0] == ctx.out_shape_vec[1] &&
                    ctx.first_strides_vec[1] == 1 &&
                    ctx.second_strides_vec[0] == 0 &&
                    ctx.second_strides_vec[1] == 1) {

                int N = ctx.out_shape_vec[0];
                int C = ctx.out_shape_vec[1];

                // #pragma omp parallel for
                for (int n = 0; n < N; ++n) {
                    // #pragma omp simd
                    for (int c = 0; c < C; ++c) {
                        result_raw[n * C + c] = first_raw[n * C + c] + second_raw[c];
                    }
                }
                handled = true;
            }
        }

        // generic fallback
        if (!handled) {
            for (int i = 0; i < result->total_count; ++i) {
                result_raw[i] =
                    first->get(i, ctx.first_strides_vec, ctx.out_shape_vec) +
                    second->get(i, ctx.second_strides_vec, ctx.out_shape_vec);
            }
        }
    } else if (ctx.device == Device::CUDA) {
        cuda::binary_op_strided('+', first->raw_values(), ctx.first_strides, second->raw_values(), ctx.second_strides,
                                ctx.out_shape, result->total_count, ctx.ndim, result->raw_values());
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), ctx](){
        if(auto r = res.lock()){
            // TODO: i dont really like that cpu is inline and cuda calls a function
            if (r->device == Device::CPU) {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, ctx.first_strides_vec, ctx.out_shape_vec, r->parents.first->grad_get(i, ctx.first_strides_vec, ctx.out_shape_vec) + r->grad_get(i));
                    r->parents.second->grad_set(i, ctx.second_strides_vec, ctx.out_shape_vec, r->parents.second->grad_get(i, ctx.second_strides_vec, ctx.out_shape_vec) + r->grad_get(i));
                }
            } else if (r->device == Device::CUDA) {
                cuda::binary_op_backward_strided('+', r->parents.first->raw_values(), ctx.first_strides,
                                               r->parents.second->raw_values(), ctx.second_strides,
                                               r->parents.first->raw_grads(), r->parents.second->raw_grads(),
                                               r->raw_grads(), ctx.out_shape, r->total_count, ctx.ndim);
            }
        }
    };
    
    return result;
}

Tensor_ptr operator-(Tensor_ptr first, Tensor_ptr second) {
    BinaryOpContext ctx(first, second);

    auto result = Tensor::init(ctx.out_shape_vec, true, ctx.device);
    result->parents = std::pair{first, second};
    result->op = "sub";

    if (ctx.device == Device::CPU) {
        for (int i = 0; i < result->total_count; ++i) {
            result->set(i, first->get(i, ctx.first_strides_vec, ctx.out_shape_vec) - second->get(i, ctx.second_strides_vec, ctx.out_shape_vec));
        }
    } else if (ctx.device == Device::CUDA) {
        cuda::binary_op_strided('-', first->raw_values(), ctx.first_strides, second->raw_values(), ctx.second_strides,
                                ctx.out_shape, result->total_count, ctx.ndim, result->raw_values());
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result), ctx](){
        if(auto r = res.lock()){
            if (r->device == Device::CPU) {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, ctx.first_strides_vec, ctx.out_shape_vec, r->parents.first->grad_get(i, ctx.first_strides_vec, ctx.out_shape_vec) + r->grad_get(i));
                    r->parents.second->grad_set(i, ctx.second_strides_vec, ctx.out_shape_vec, r->parents.second->grad_get(i, ctx.second_strides_vec, ctx.out_shape_vec) - r->grad_get(i));
                }
            } else if (r->device == Device::CUDA) {
                cuda::binary_op_backward_strided('-', r->parents.first->raw_values(), ctx.first_strides,
                                               r->parents.second->raw_values(), ctx.second_strides,
                                               r->parents.first->raw_grads(), r->parents.second->raw_grads(),
                                               r->raw_grads(), ctx.out_shape, r->total_count, ctx.ndim);
            }
        }
    };
    return result;
}

Tensor_ptr operator*(Tensor_ptr first, Tensor_ptr second) {
    BinaryOpContext ctx(first, second);

    auto result = Tensor::init(ctx.out_shape_vec, true, ctx.device);
    result->parents = std::pair{first, second};
    result->op = "mul";

    if (ctx.device == Device::CPU) {
        for (int i = 0; i < result->total_count; ++i) {
            result->set(i, first->get(i, ctx.first_strides_vec, ctx.out_shape_vec) * second->get(i, ctx.second_strides_vec, ctx.out_shape_vec));
        }
    } else if (ctx.device == Device::CUDA) {
        cuda::binary_op_strided('*', first->raw_values(), ctx.first_strides, second->raw_values(), ctx.second_strides,
                                ctx.out_shape, result->total_count, ctx.ndim, result->raw_values());
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result), ctx](){
        if(auto r = res.lock()){
            if (r->device == Device::CPU) {
                for (int i = 0; i < r->total_count; ++i) {
                    r->parents.first->grad_set(i, ctx.first_strides_vec, ctx.out_shape_vec, r->parents.first->grad_get(i, ctx.first_strides_vec, ctx.out_shape_vec) + r->grad_get(i) * r->parents.second->get(i, ctx.second_strides_vec, ctx.out_shape_vec));
                    r->parents.second->grad_set(i, ctx.second_strides_vec, ctx.out_shape_vec, r->parents.second->grad_get(i, ctx.second_strides_vec, ctx.out_shape_vec) + r->grad_get(i) * r->parents.first->get(i, ctx.first_strides_vec, ctx.out_shape_vec));
                }
            } else if (r->device == Device::CUDA) {
                cuda::binary_op_backward_strided('*', r->parents.first->raw_values(), ctx.first_strides,
                                               r->parents.second->raw_values(), ctx.second_strides,
                                               r->parents.first->raw_grads(), r->parents.second->raw_grads(),
                                               r->raw_grads(), ctx.out_shape, r->total_count, ctx.ndim);
            }
        }
    };
    return result;
}

Tensor_ptr operator/(Tensor_ptr first, Tensor_ptr second) {
    BinaryOpContext ctx(first, second);

    auto result = Tensor::init(ctx.out_shape_vec, true, ctx.device);
    result->parents = std::pair{first, second};
    result->op = "div";

    if (ctx.device == Device::CPU) {
        for (int i = 0; i < result->total_count; ++i) {
            result->set(i, first->get(i, ctx.first_strides_vec, ctx.out_shape_vec) / second->get(i, ctx.second_strides_vec, ctx.out_shape_vec));
        }
    } else if (ctx.device == Device::CUDA) {
        cuda::binary_op_strided('/', first->raw_values(), ctx.first_strides, second->raw_values(), ctx.second_strides,
                                ctx.out_shape, result->total_count, ctx.ndim, result->raw_values());
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), ctx](){
        if(auto r = res.lock()){
            if (r->device == Device::CPU) {
                for (int i = 0; i < r->total_count; ++i) {
                    float s_val = r->parents.second->get(i, ctx.second_strides_vec, ctx.out_shape_vec);
                    r->parents.first->grad_set(i, ctx.first_strides_vec, ctx.out_shape_vec, r->parents.first->grad_get(i, ctx.first_strides_vec, ctx.out_shape_vec) + r->grad_get(i) * (1.0f / s_val));
                    r->parents.second->grad_set(i, ctx.second_strides_vec, ctx.out_shape_vec, r->parents.second->grad_get(i, ctx.second_strides_vec, ctx.out_shape_vec) + r->grad_get(i) * 
                        -(r->parents.first->get(i, ctx.first_strides_vec, ctx.out_shape_vec) / (s_val * s_val)));
                }
            } else if (r->device == Device::CUDA) {
                cuda::binary_op_backward_strided('/', r->parents.first->raw_values(), ctx.first_strides,
                                               r->parents.second->raw_values(), ctx.second_strides,
                                               r->parents.first->raw_grads(), r->parents.second->raw_grads(),
                                               r->raw_grads(), ctx.out_shape, r->total_count, ctx.ndim);
            }
        }
    };
    return result;
}

Tensor_ptr Tensor::matmul(Tensor_ptr tensor) {
    if (shape.size() != 2 || tensor->shape.size() != 2) throw std::invalid_argument("Matmul defined only for 2d tensors");
    if (shape[1] != tensor->shape[0]) throw std::invalid_argument("Invalid shapes for matmul");
    if (device != tensor->device) throw std::invalid_argument("different devices");
    // TODO: fix
    if (device == Device::CUDA && (!is_continous() || !tensor->is_continous())) throw std::invalid_argument("CUDA only for continous for now");

    Tensor_ptr result = Tensor::init({shape[0], tensor->shape[1]}, true, device);
    result->parents = std::pair{shared_from_this(), tensor};
    result->op = "matmul";
    
    if (device == Device::CPU) {
        float* result_data = result->raw_values();
        if (is_continous() && tensor->is_continous()) {
            cpu::matmul(raw_values(), tensor->raw_values(), result_data, shape[1], tensor->shape[1], shape[0]);
        }
        else {
            std::vector<float> v1 = values_vec();
            std::vector<float> v2 = tensor->values_vec();
            cpu::matmul(v1.data(), v2.data(), result_data, shape[1], tensor->shape[1], shape[0]);
        }
    } else if (device == Device::CUDA) {
        // TODO: temporary so tests works
        auto matmul_output = cuda::matmul(values_vec(), tensor->values_vec(), shape[1], tensor->shape[1], shape[0]);
        for (int i = 0; i < matmul_output.size(); ++i)
            result->set(i, matmul_output[i]);
    }
    
    result->backward_fn = [res = std::weak_ptr<Tensor>(result)](){
        if(auto r = res.lock()){
            if (r->device == Device::CPU) {
                if (r->parents.first->is_continous() && r->parents.second->is_continous()) {
                    float* grad_first = new float[r->parents.first->total_count];
                    float* grad_second = new float[r->parents.second->total_count];
                    cpu::BT_matmul(r->raw_grads(), r->parents.second->raw_values(), grad_first, r->shape[1], r->shape[0], r->parents.second->shape[0]);
                    
                    int m = r->parents.first->shape[0];
                    int k = r->parents.first->shape[1];
                    int n = r->shape[1];
                    cpu::AT_matmul(r->parents.first->raw_values(), r->raw_grads(), grad_second, m, k, n);
                    
                    float* raw_grad_first = r->parents.first->raw_grads();
                    for (int i = 0; i < r->parents.first->total_count; ++i) {
                        raw_grad_first[i] += grad_first[i];
                    }
                    
                    float* raw_grad_second = r->parents.second->raw_grads();
                    for (int i = 0; i < r->parents.second->total_count; ++i) {
                        raw_grad_second[i] += grad_second[i];
                    }

                    delete[] grad_first;
                    delete[] grad_second;
                }
                else {
                    printf("slow...\n");
                    // TODO: its ugly
                    Tensor_ptr firstT = r->parents.first->transpose();

                    std::vector<float> grad_first(r->parents.first->total_count);
                    std::vector<float> g_vec = r->grads_vec();
                    std::vector<float> p2_vec = r->parents.second->values_vec();
                    cpu::BT_matmul(g_vec.data(), p2_vec.data(), grad_first.data(), r->shape[1], r->shape[0], r->parents.second->shape[0]);

                    std::vector<float> grad_second(r->parents.second->total_count);
                    std::vector<float> fT_vec = firstT->values_vec();
                    cpu::matmul(fT_vec.data(), g_vec.data(), grad_second.data(), firstT->shape[1], r->shape[1], firstT->shape[0]);
               
                    if (r->parents.first->is_continous()) {
                        float* raw_grad = r->parents.first->raw_grads();
                        int count = r->parents.first->total_count;
                        for (int i = 0; i < count; ++i) {
                            raw_grad[i] += grad_first[i];
                        }
                    }
                    else {
                        for (int i = 0; i < r->parents.first->total_count; ++i) {
                            r->parents.first->grad_set(i, r->parents.first->grad_get(i) + grad_first[i]);
                        }
                    }
                    
                    if (r->parents.second->is_continous()) {
                        float* raw_grad = r->parents.second->raw_grads();
                        int count = r->parents.second->total_count;
                        for (int i = 0; i < count; ++i) {
                            raw_grad[i] += grad_second[i];
                        }
                    }
                    else {
                        for (int i = 0; i < r->parents.second->total_count; ++i) {
                            r->parents.second->grad_set(i, r->parents.second->grad_get(i) + grad_second[i]);
                        }
                    }
                }
            } else if (r->device == Device::CUDA) {
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

    if (device == Device::CPU) {
        if (is_continous()) cpu::softmax(raw_values(), result->values.get(), N, C);
        else {
            std::vector<float> v = values_vec();
            cpu::softmax(v.data(), result->values.get(), N, C);
        }
    } else if (device == Device::CUDA) {
        cuda::make_continous(shared_from_this());
        cuda::softmax(values.get(), result->values.get(), N, C);
    }

    result->backward_fn = [res = std::weak_ptr<Tensor>(result), N, C](){
        if(auto r = res.lock()){
            if (r->device == Device::CPU) {
                for (int n = 0; n < N; ++n) {
                    float sum = 0.0;
                    for (int c = 0; c < C; ++c) {
                        sum += r->grad_get({n, c}) * r->get({n, c});
                    }
                    for (int c1 = 0; c1 < C; ++c1) {
                        r->parents.first->grad_set({n, c1}, r->parents.first->grad_get({n, c1}) + r->get({n, c1}) * (r->grad_get({n, c1}) - sum));
                    }
                }
            } else if (r->device == Device::CUDA) {
                cuda::softmax_backward(r->values.get(), r->parents.first->grads.get(), r->grads.get(), N, C);
            }
        }
    };

    return result;
}

Tensor_ptr Tensor::transpose()
{
    Tensor_ptr result = std::make_shared<Tensor>();
    result->values = values;
    result->grads = grads;
    result->total_count = total_count;
    result->shape = shape;
    result->strides = strides;
    result->device = device;
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "transpose";
    std::ranges::reverse(result->shape);
    std::ranges::reverse(result->strides);
    return result;
}

Tensor_ptr Tensor::view(std::vector<int> new_shape)
{
    int new_total_count = 1;
    for (auto s : new_shape) new_total_count *= s;
    if (new_total_count != total_count) throw std::invalid_argument("View shape must have same total count");

    Tensor_ptr result = std::make_shared<Tensor>();
    result->values = values;
    result->grads = grads;
    result->total_count = total_count;
    result->shape = new_shape;
    result->strides = stride::calc_strides(new_shape);;
    result->device = device;
    result->parents = std::pair{shared_from_this(), nullptr};
    result->op = "view";
    return result;
}

void Tensor::backward() {
    if (total_count != 1) throw std::invalid_argument("backward only possible on 1x1 tensor");
    std::unordered_set<Tensor_ptr> visited{};
    std::vector<Tensor_ptr> topo{};
    toposort(shared_from_this(), visited, topo);
    std::ranges::reverse(topo);
    grads.set(0, 1.0f);

    for (auto const& t : topo) {
        if (t->backward_fn) {
            t->backward_fn();
        }
    }
}

void Tensor::toposort(const Tensor_ptr& t, std::unordered_set<Tensor_ptr>& visited, std::vector<Tensor_ptr>& res) {
    if (!visited.contains(t)) {
        visited.insert(t);
        if (t->parents.first)
            toposort(t->parents.first, visited, res);
        if (t->parents.second)
            toposort(t->parents.second, visited, res);
        res.push_back(t);
    }
}

void Tensor::zero_grad() {
    if (device == Device::CPU) {
        for (int i = 0; i < total_count; ++i) grad_set(i, 0);
    } else if (device == Device::CUDA) {
        cudaMemset(grads.get(), 0, total_count * sizeof(float));
    }
    if (parents.first && parents.first.get() != this)   parents.first->zero_grad();
    if (parents.second && parents.second.get() != this) parents.second->zero_grad();
}

Tensor_ptr Tensor::im2col(int kernel_size, int stride, int padding) {
    if (shape.size() != 4) throw std::invalid_argument("im2col defined only for 4d tensors");
    if (!is_continous()) throw std::invalid_argument("only for continous");
    int batch = shape[0];
    int height = shape[1];
    int width = shape[2];
    int channels = shape[3];
    int out_h = (height - kernel_size + 2 * padding) / stride + 1;
    int out_w = (width  - kernel_size + 2 * padding) / stride + 1;

    Tensor_ptr result = Tensor::init({batch, out_h, out_w, kernel_size * kernel_size * channels}, true, device);
    if (device == Device::CPU) cpu::im2col(raw_values(), result->raw_values(), batch, height, width, out_h, out_w, channels, kernel_size, stride, padding);
    if (device == Device::CUDA) cuda::im2col(raw_values(), result->raw_values(), batch, height, width, out_h, out_w, channels, kernel_size, stride, padding);
    
    result->image = true;

    return result;
}

Tensor_ptr Tensor::im2col_naive(int kernel_size, int stride, int padding) {
    if (shape.size() != 4) throw std::invalid_argument("im2col defined only for 4d tensors");
    int batch = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];

    int out_h = (height - kernel_size + 2 * padding) / stride + 1;
    int out_w = (width  - kernel_size + 2 * padding) / stride + 1;
    Tensor_ptr result = Tensor::init({batch, out_h * out_w, kernel_size * kernel_size * channels}, true, device);

    // #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                int x = ox * stride - padding;
                int y = oy * stride - padding;
                int col = oy * out_w + ox;

                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        for (int c = 0; c < channels; ++c) {
                            int row = c * kernel_size * kernel_size + ky * kernel_size + kx;

                            int in_y = y + ky;
                            int in_x = x + kx;

                            float val = 0;

                            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width)
                                val = get({b, c, in_y, in_x});

                            result->set({b, col, row}, val);
                        }
                    }
                }
            }
        }
    }

    return result;
}
