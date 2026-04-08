#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <cuda/cmath>
#include <vector>
#include <stdexcept>
#include <mma.h>
#include <cuda_fp16.h>
#include <array>

#include "globals.hpp"
#include "cuda_kernels.h"

#ifdef CUDA_TEST
#include "cuda_debug.h"
int g_cuda_kernel_launches = 0;
#endif

struct LayoutMeta {
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int dims;
};

// ----------------

template <char Op>
__global__ void binary_op_kernel(const float* input_A, const float* input_B, float* output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        if constexpr (Op == '+') output[idx] = input_A[idx] + input_B[idx];
        else if constexpr (Op == '-') output[idx] = input_A[idx] - input_B[idx];
        else if constexpr (Op == '*') output[idx] = input_A[idx] * input_B[idx];
        else if constexpr (Op == '/') output[idx] = input_A[idx] / input_B[idx];
    }
}

void launch_binary_op(const char op, const float* input_A, const float* input_B, float* output, int size) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    switch (op) {
        case '+': binary_op_kernel<'+'><<<blocks, threads>>>(input_A, input_B, output, size); break;
        case '-': binary_op_kernel<'-'><<<blocks, threads>>>(input_A, input_B, output, size); break;
        case '*': binary_op_kernel<'*'><<<blocks, threads>>>(input_A, input_B, output, size); break;
        case '/': binary_op_kernel<'/'><<<blocks, threads>>>(input_A, input_B, output, size); break;
        default: throw std::invalid_argument("Unknown op");
    }
}

__device__ __forceinline__ int strided_idx_device(int idx, const int strides[MAX_DIMS], const int shape[MAX_DIMS], int dims) {
    int res = 0;
    for (int i = dims - 1; i >= 0; --i) {
        int coord = idx % shape[i];
        idx = idx / shape[i];
        res += coord * strides[i];
    }
    return res;
}

struct BinaryOpMeta {
    int strides_A[MAX_DIMS];
    int strides_B[MAX_DIMS];
    int shape[MAX_DIMS];
    int dims;
    int size;
};

template <char Op>
__global__ void binary_op_strided_kernel(const float* input_A, const float* input_B, float* output, BinaryOpMeta meta) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < meta.size) {
        int idx_a = strided_idx_device(idx, meta.strides_A, meta.shape, meta.dims);
        int idx_b = strided_idx_device(idx, meta.strides_B, meta.shape, meta.dims);
        if constexpr (Op == '+') output[idx] = input_A[idx_a] + input_B[idx_b];
        else if constexpr (Op == '-') output[idx] = input_A[idx_a] - input_B[idx_b];
        else if constexpr (Op == '*') output[idx] = input_A[idx_a] * input_B[idx_b];
        else if constexpr (Op == '/') output[idx] = input_A[idx_a] / input_B[idx_b];
    }
}

// TODO: fix this std::array mess
void launch_binary_op_strided(const char op, const float* input_A, std::array<int, MAX_DIMS> strides_A,
                              const float* input_B, std::array<int, MAX_DIMS> strides_B, std::array<int, MAX_DIMS> shape,
                              float* output, int size, int dims) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    BinaryOpMeta meta;
    meta.size = size;
    meta.dims = dims;
    for (int i = 0; i < dims; ++i) {
        meta.strides_A[i] = strides_A[i];
        meta.strides_B[i] = strides_B[i];
        meta.shape[i] = shape[i];
    }

    switch (op) {
        case '+': binary_op_strided_kernel<'+'><<<blocks, threads>>>(input_A, input_B, output, meta); break;
        case '-': binary_op_strided_kernel<'-'><<<blocks, threads>>>(input_A, input_B, output, meta); break;
        case '*': binary_op_strided_kernel<'*'><<<blocks, threads>>>(input_A, input_B, output, meta); break;
        case '/': binary_op_strided_kernel<'/'><<<blocks, threads>>>(input_A, input_B, output, meta); break;
        default: throw std::invalid_argument("Unknown op");
    }
}

template <char Op>
__global__ void binary_op_backward_strided_kernel(const float* input_A, const float* input_B, float* grad_A, float* grad_B, const float* grad_output, BinaryOpMeta meta) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < meta.size) {
        int idx_a = strided_idx_device(idx, meta.strides_A, meta.shape, meta.dims);
        int idx_b = strided_idx_device(idx, meta.strides_B, meta.shape, meta.dims);
        
        float g_out = grad_output[idx];
        
        if constexpr (Op == '+') {
            atomicAdd(&grad_A[idx_a], g_out);
            atomicAdd(&grad_B[idx_b], g_out);
        } else if constexpr (Op == '-') {
            atomicAdd(&grad_A[idx_a], g_out);
            atomicAdd(&grad_B[idx_b], -g_out);
        } else if constexpr (Op == '*') {
            atomicAdd(&grad_A[idx_a], g_out * input_B[idx_b]);
            atomicAdd(&grad_B[idx_b], g_out * input_A[idx_a]);
        } else if constexpr (Op == '/') {
            atomicAdd(&grad_A[idx_a], g_out / input_B[idx_b]);
            atomicAdd(&grad_B[idx_b], -g_out * input_A[idx_a] / (input_B[idx_b] * input_B[idx_b]));
        }
    }
}

void launch_binary_op_backward_strided(const char op, const float* input_A, std::array<int, MAX_DIMS> strides_A,
                                      const float* input_B, std::array<int, MAX_DIMS> strides_B,
                                      float* grad_A, float* grad_B, const float* grad_output,
                                      std::array<int, MAX_DIMS> shape, int size, int dims) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    BinaryOpMeta meta;
    meta.size = size;
    meta.dims = dims;
    for (int i = 0; i < dims; ++i) {
        meta.strides_A[i] = strides_A[i];
        meta.strides_B[i] = strides_B[i];
        meta.shape[i] = shape[i];
    }

    switch (op) {
        case '+': binary_op_backward_strided_kernel<'+'><<<blocks, threads>>>(input_A, input_B, grad_A, grad_B, grad_output, meta); break;
        case '-': binary_op_backward_strided_kernel<'-'><<<blocks, threads>>>(input_A, input_B, grad_A, grad_B, grad_output, meta); break;
        case '*': binary_op_backward_strided_kernel<'*'><<<blocks, threads>>>(input_A, input_B, grad_A, grad_B, grad_output, meta); break;
        case '/': binary_op_backward_strided_kernel<'/'><<<blocks, threads>>>(input_A, input_B, grad_A, grad_B, grad_output, meta); break;
        default: throw std::invalid_argument("Unknown op");
    }
}

// TODO: could be faster https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <ReductionOp op>
__global__ void reduction_kernel(const float* input, float* output, int size) {
    extern __shared__ float shared_array[];

    int b_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int t_idx = threadIdx.x;

    shared_array[t_idx] = b_idx < size? input[b_idx]: 0;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (t_idx < i) {
            if constexpr (op == ReductionOp::MAX) {
                if (shared_array[t_idx] < shared_array[t_idx + i]) shared_array[t_idx] = shared_array[t_idx + i];
            }
            if constexpr (op == ReductionOp::MIN) {
                if (shared_array[t_idx] > shared_array[t_idx + i]) shared_array[t_idx] = shared_array[t_idx + i];
            }
            if constexpr (op == ReductionOp::SUM) {
                shared_array[t_idx] += shared_array[t_idx + i];
            }
        }
        __syncthreads();
    }

    if (t_idx == 0) output[blockIdx.x] = shared_array[0];
}

int launch_reduction(const ReductionOp op, const float* input, float* output, int size) {
#ifdef CUDA_TEST 
    g_cuda_kernel_launches++; 
#endif 

    constexpr int block = 256; 
    int grid = cuda::ceil_div(size, block);
    int shared_memory = sizeof(float) * block; 
    switch (op) {
        case ReductionOp::MAX: 
            reduction_kernel<ReductionOp::MAX><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        case ReductionOp::MIN: 
            reduction_kernel<ReductionOp::MIN><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        case ReductionOp::SUM: 
            reduction_kernel<ReductionOp::SUM><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        default: throw std::invalid_argument("Unknown op");
    }

    return grid;
}

// ----------------------------------

template <ReductionOp op>
__global__ void reduction_kernel2(const float* input, float* output, int size)
{
    extern __shared__ float shared_array[];
    int t_idx = threadIdx.x;
    
    float local_max = -DBL_MAX;
    for (int i = t_idx; i < size; i += blockDim.x)
        local_max = fmax(local_max, input[i]);

    shared_array[t_idx] = local_max;
    
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if constexpr (op == ReductionOp::MAX) {
            if (t_idx < i && shared_array[t_idx] < shared_array[t_idx + i])
                shared_array[t_idx] = shared_array[t_idx + i];
        }
        __syncthreads();
    }

    if (t_idx == 0) *output = shared_array[0];
}

void launch_full_reduction(const ReductionOp op, const float* input, float* output, int size) {
#ifdef CUDA_TEST 
    g_cuda_kernel_launches++; 
#endif 

    constexpr int block = 256; 
    int grid = cuda::ceil_div(size, block);
    int shared_memory = sizeof(float) * block; 
    switch (op) {
        case ReductionOp::MAX: 
            reduction_kernel2<ReductionOp::MAX><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        default: throw std::invalid_argument("Unknown op");
    }
}


__global__ void softmax_kernel2(const float* input, float* output, int N, int C) {
    extern __shared__ float shared_array[];
    int n = blockIdx.x;
    int c = threadIdx.x;
    
    bool active = c < C;
    shared_array[c] = active? input[n * C + c]: -DBL_MAX;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (c < i && c + i < C && shared_array[c] < shared_array[c + i])
            shared_array[c] = shared_array[c + i];
        __syncthreads();
    }

    float max_val = shared_array[0];
    // printf("%d: %f\n", c, max_val);
    __syncthreads();

    float exp_val;
    if (active) {
        exp_val = exp(input[n * C + c] - max_val);
        shared_array[c] = exp_val;
        // printf("%d: %f\n", c, exp_val);
    }
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (c < i && c + i < C) {
            // printf("%d: %f += %f\n", c, shared_array[c], shared_array[c + i]);
            shared_array[c] += shared_array[c + i];
        }
        __syncthreads();
    }
    
    if (active) {
        // printf("%d: %f / %f\n", c, exp_val, shared_array[0]);
        output[n * C + c] = exp_val / shared_array[0];
    }
}

void launch_softmax2(const float* input, float* output, int N, int C) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    // one row per block
    int block = (C + 32 - 1) / 32 * 32;
    size_t shared_memory_bytes = block * sizeof(float);
    int grid = N;

    softmax_kernel2<<<grid, block, shared_memory_bytes>>>(input, output, N, C);
}

// -----------------------------------

__global__ void apply_strided_idx(const float* input, float* output, int count, LayoutMeta meta) {
    int strided_idx = 0, temp;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int current_idx = output_idx;
    if (current_idx >= count) return;

    for (int i = meta.dims - 1; i >= 0; --i) {
            temp = current_idx % meta.shape[i];
            current_idx = current_idx / meta.shape[i];
            strided_idx += temp * meta.strides[i];
        }

    output[output_idx] = input[strided_idx];
}

void launch_make_continous(const float* input, float* output, int count, const std::vector<int> &strides, const std::vector<int> &shape) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    constexpr int block = 256;
    int grid = cuda::ceil_div(count, block);

    LayoutMeta meta = {};
    meta.dims = shape.size();

    for (int i = 0; i < meta.dims; ++i) {
        meta.shape[i] = shape[i];
        meta.strides[i] = strides[i];
    }

    apply_strided_idx<<<grid, block>>>(input, output, count, meta);
}

// ----------------------

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

void launch_relu(const float* input, float* output, int size) {
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    relu_kernel<<<blocks, threads>>>(input, output, size);
}

__global__ void exp_kernel(const float* input, float* output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        output[idx] = exp(input[idx]);
    }
}

void launch_exp(const float* input, float* output, int size) {
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    exp_kernel<<<blocks, threads>>>(input, output, size);
}

// ----------------------
// Backward kernels

__global__ void relu_backward_kernel(const float* input, float* grad_input, const float* grad_output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] += (input[idx] > 0 ? 1.0f : 0.0f) * grad_output[idx];
    }
}

void launch_relu_backward(const float* input, float* grad_input, const float* grad_output, int size) {
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    relu_backward_kernel<<<blocks, threads>>>(input, grad_input, grad_output, size);
}

__global__ void sum_backward_kernel(float* grad_input, const float* grad_output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] += grad_output[0];
    }
}

void launch_sum_backward(float* grad_input, const float* grad_output, int size) {
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    sum_backward_kernel<<<blocks, threads>>>(grad_input, grad_output, size);
}

__global__ void sum_axis_backward_kernel(float* grad_input, const float* grad_output, int N, int C, int axis) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N * C) {
        int n = idx / C;
        int c = idx % C;
        grad_input[idx] += grad_output[axis == 0 ? c : n];
    }
}

void launch_sum_axis_backward(float* grad_input, const float* grad_output, int N, int C, int axis) {
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(N * C, threads);
    sum_axis_backward_kernel<<<blocks, threads>>>(grad_input, grad_output, N, C, axis);
}

__global__ void exp_backward_kernel(const float* output, float* grad_input, const float* grad_output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] += grad_output[idx] * output[idx];
    }
}

void launch_exp_backward(const float* output, float* grad_input, const float* grad_output, int size) {
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    exp_backward_kernel<<<blocks, threads>>>(output, grad_input, grad_output, size);
}

__global__ void softmax_backward_kernel(const float* output, float* grad_input, const float* grad_output, int N, int C) {
    extern __shared__ float shared_array[];
    int n = blockIdx.x;
    int c = threadIdx.x;
    bool active = c < C;

    float sum = 0.0f;
    if (active) {
        sum = grad_output[n * C + c] * output[n * C + c];
    }
    shared_array[c] = sum;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (c < i && c + i < C) {
            shared_array[c] += shared_array[c + i];
        }
        __syncthreads();
    }

    float total_sum = shared_array[0];
    if (active) {
        grad_input[n * C + c] += output[n * C + c] * (grad_output[n * C + c] - total_sum);
    }
}

void launch_softmax_backward(const float* output, float* grad_input, const float* grad_output, int N, int C) {
    int block = (C + 31) / 32 * 32;
    size_t shared_memory_bytes = block * sizeof(float);
    int grid = N;
    softmax_backward_kernel<<<grid, block, shared_memory_bytes>>>(output, grad_input, grad_output, N, C);
}

// end backward kernels
// -------

__global__ void transpose_kernel(float* matrix, float* matrixT, int N, int C) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N * C) {
        int n = idx / C;
        int c = idx % C;

        matrixT[c * N + n] = matrix[idx];
    }
}

void launch_transpose(float* matrix, float* matrixT, int N, int C) {
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(N * C, threads);
    transpose_kernel<<<blocks, threads>>>(matrix, matrixT, N, C);
}

__global__ void im2col(const float* __restrict__ in_data, float* __restrict__ res_data, int batches, int height, int width, int out_h, int out_w, int channels, int kernel_size, int stride, int padding) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batches * out_h * out_w;

    if (idx >= total) return;

    int ox = idx % out_w;
    int oy = (idx / out_w) % out_h;
    int b = idx / (out_w * out_h);

    int flatten_kernel = kernel_size * kernel_size * channels;

    int in_stride_b = height * width * channels;
    int in_stride_h = width * channels;

    int res_stride_b = out_h * out_w * flatten_kernel;
    int res_stride_h = out_w * flatten_kernel;

    int kernel_size2 = kernel_size * kernel_size;

    int x = ox * stride - padding;
    int y = oy * stride - padding;

    for (int c = 0; c < channels; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int row = c * kernel_size2 + ky * kernel_size + kx;

                int in_y = y + ky;
                int in_x = x + kx;

                float val = 0;

                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    int in_idx = b * in_stride_b + in_y * in_stride_h + in_x * channels + c;
                    val = in_data[in_idx];
                }

                int res_idx = b * res_stride_b + oy * res_stride_h + ox * flatten_kernel + row;
                res_data[res_idx] = val;
            }
        }
    }
}

void launch_im2col(const float* __restrict__ in_data, float* __restrict__ res_data, int batches, int height, int width, int out_h, int out_w, 
    int channels, int kernel_size, int stride, int padding) {

    #ifdef CUDA_TEST
    g_cuda_kernel_launches++;
    #endif
    
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(batches * out_h * out_w, threads);
    im2col<<<blocks, threads>>>(in_data, res_data, batches, height, width, out_h, out_w, channels, kernel_size, stride, padding);
}