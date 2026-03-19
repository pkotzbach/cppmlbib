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
    int threads = 256;
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
    int threads = 256;
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
    int threads = 256;
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

// -------------------

__global__ void matmul_kernel_naive(const float* A, const float* B, float* C, int K, int X, int Y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < Y && x < X) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[y * K + k] * B[k * X + x];
        }
        C[y * X + x] = sum;
    }
}

void launch_matmul_naive(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y)
{
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif

    dim3 block(16, 16);
    dim3 grid(cuda::ceil_div(X, block.x), cuda::ceil_div(Y, block.y));

    matmul_kernel_naive<<<grid, block>>>(d_A, d_B, d_C, K, X, Y);
}


#define PER_THREAD 8
#define BLOCK_K 8
#define BLOCK_X 64
#define BLOCK_Y 64
__global__ void matmul_kernel(const float* A, const float* B, float* C, int K, int X, int Y) {
    __shared__ float sA[BLOCK_Y * BLOCK_K];
    __shared__ float sB[BLOCK_K * BLOCK_X];

    const int block_steps = BLOCK_X / PER_THREAD; // or BLOCK_Y
    
    const int thread_x = threadIdx.x % block_steps;
    const int thread_y = threadIdx.x / block_steps;

    const int VEC_SIZE = 4; // float4
    const int NUM_THREADS = (BLOCK_X / PER_THREAD) * (BLOCK_Y / PER_THREAD);

    const int SA_VECS_PER_ROW = BLOCK_K / VEC_SIZE;
    const int SA_ROWS_PER_STEP = NUM_THREADS / SA_VECS_PER_ROW;
    const int SA_STEPS = (BLOCK_Y * BLOCK_K) / (VEC_SIZE * NUM_THREADS);

    const int SB_VECS_PER_ROW = BLOCK_X / VEC_SIZE;
    const int SB_ROWS_PER_STEP = NUM_THREADS / SB_VECS_PER_ROW;
    const int SB_STEPS = (BLOCK_K * BLOCK_X) / (VEC_SIZE * NUM_THREADS);

    const int a_row = threadIdx.x / SA_VECS_PER_ROW;
    const int a_col = (threadIdx.x % SA_VECS_PER_ROW) * VEC_SIZE;
    const int b_row = threadIdx.x / SB_VECS_PER_ROW;
    const int b_col = (threadIdx.x % SB_VECS_PER_ROW) * VEC_SIZE;

    float sum[PER_THREAD * PER_THREAD] = {0};

    float regA[PER_THREAD] = {0};
    float regB[PER_THREAD] = {0};

    for (int k = 0; k < K; k += BLOCK_K) {

        for (int step = 0; step < SA_STEPS; ++step) {
            int a_r = a_row + step * SA_ROWS_PER_STEP;
            int global_a_r = blockIdx.y * BLOCK_Y + a_r;
            int global_a_c = k + a_col;

            if (global_a_r < Y && global_a_c < K) {
                // sA is transposed
                reinterpret_cast<float4*>(&sA[a_r * BLOCK_K + a_col])[0] = reinterpret_cast<const float4*>(&A[global_a_r * K + global_a_c])[0];
            } else {
                sA[a_r * BLOCK_K + a_col + 0] = 0;
                sA[a_r * BLOCK_K + a_col + 1] = 0;
                sA[a_r * BLOCK_K + a_col + 2] = 0;
                sA[a_r * BLOCK_K + a_col + 3] = 0;
            }
        }

        for (int step = 0; step < SB_STEPS; ++step) {
            int b_r = b_row + step * SB_ROWS_PER_STEP;
            int global_b_r = k + b_r;
            int global_b_c = blockIdx.x * BLOCK_X + b_col;

            if (global_b_r < K && global_b_c < X) {
                reinterpret_cast<float4*>(&sB[b_r * BLOCK_X + b_col])[0] = reinterpret_cast<const float4*>(&B[global_b_r * X + global_b_c])[0];
            } else {
                sB[b_r * BLOCK_X + b_col + 0] = 0;
                sB[b_r * BLOCK_X + b_col + 1] = 0;
                sB[b_r * BLOCK_X + b_col + 2] = 0;
                sB[b_r * BLOCK_X + b_col + 3] = 0;
            }
        }

        __syncthreads();

        // inner loop
        for (int k_in = 0; k_in < BLOCK_K; ++k_in) {
            for (int i = 0; i < PER_THREAD; ++i) {
                regA[i] = sA[(thread_y * PER_THREAD + i) * BLOCK_K + k_in];
                regB[i] = sB[k_in * BLOCK_X + thread_x * PER_THREAD + i];
            }

            for (int i = 0; i < PER_THREAD; ++i) {
                for (int j = 0; j < PER_THREAD; ++j) {
                    sum[i * PER_THREAD + j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    for (int y = 0; y < PER_THREAD; ++y) {
        int C_y = blockIdx.y * BLOCK_Y + thread_y * PER_THREAD + y;
        if (C_y >= Y) continue;

        for (int x = 0; x < PER_THREAD; ++x) {
            int C_x = blockIdx.x * BLOCK_X + thread_x * PER_THREAD + x;
            if (C_x < X) {
                C[C_y * X + C_x] = sum[y * PER_THREAD + x];
            }
        }
    }
}

void launch_matmul(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y)
{
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    static_assert(BLOCK_X == BLOCK_Y);
    static_assert(BLOCK_Y / PER_THREAD == PER_THREAD);
    dim3 grid(cuda::ceil_div(X, BLOCK_X), cuda::ceil_div(Y, BLOCK_Y));
    dim3 block(BLOCK_X / PER_THREAD * BLOCK_Y / PER_THREAD);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, K, X, Y);
}

// ----------------------------------

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

    int block = 256; 
    int grid = cuda::ceil_div(size, block);
    int shared_memory = sizeof(float) * block; 
    switch (op) {
        case MAX: 
            reduction_kernel<MAX><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        case MIN: 
            reduction_kernel<MIN><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        case SUM: 
            reduction_kernel<SUM><<<grid, block, shared_memory>>>(input, output, size); 
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

    int block = 256; 
    int grid = cuda::ceil_div(size, block);
    int shared_memory = sizeof(float) * block; 
    switch (op) {
        case ReductionOp::MAX: 
            reduction_kernel2<ReductionOp::MAX><<<grid, block, shared_memory>>>(input, output, size); 
            break;
        default: throw std::invalid_argument("Unknown op");
    }
}

#define WMMA_Y 16
#define WMMA_X 16
#define WMMA_K 16

__global__ void convert_fp32_to_fp16(const float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = __float2half(in[idx]);
}

__global__ void matmul_tc_kernel(const half* A, const half* B, float* C, int K, int X, int Y) {
    int warp_idx_x = (threadIdx.x / 32) % 2;
    int warp_idx_y = (threadIdx.x / 32) / 2;

    int y = (blockIdx.y * 2 + warp_idx_y) * WMMA_Y;
    int x = (blockIdx.x * 2 + warp_idx_x) * WMMA_X;

    if (y >= Y || x >= X) return;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_Y, WMMA_X, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_Y, WMMA_X, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_Y, WMMA_X, WMMA_K, float> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        nvcuda::wmma::load_matrix_sync(a_frag, A + y * K + k, K);
        nvcuda::wmma::load_matrix_sync(b_frag, B + k * X + x, X);

        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    nvcuda::wmma::store_matrix_sync(C + y * X + x, acc_frag, X, nvcuda::wmma::mem_row_major);
}

void launch_matmul_tc(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y) {
    if (K % 16 != 0 || X % 16 != 0 || Y % 16 != 0) {
        launch_matmul(d_A, d_B, d_C, K, X, Y);
        return;
    }

    half *d_A_half, *d_B_half;
    cudaMalloc(&d_A_half, Y * K * sizeof(half));
    cudaMalloc(&d_B_half, K * X * sizeof(half));

    int threads = 256;
    convert_fp32_to_fp16<<<cuda::ceil_div(Y * K, threads), threads>>>(d_A, d_A_half, Y * K);
    convert_fp32_to_fp16<<<cuda::ceil_div(K * X, threads), threads>>>(d_B, d_B_half, K * X);

    dim3 block(128); 
    dim3 grid(cuda::ceil_div(X, 32), cuda::ceil_div(Y, 32));

    matmul_tc_kernel<<<grid, block>>>(d_A_half, d_B_half, d_C, K, X, Y);

    cudaFree(d_A_half);
    cudaFree(d_B_half);
}

// -----------------

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

void launch_make_continous(const float* input, float* output, int count, std::vector<int> &strides, std::vector<int> &shape) {
#ifdef CUDA_TEST
    g_cuda_kernel_launches++;
#endif
    int block = 256;
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
    int threads = 256;
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
    int threads = 256;
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
    int threads = 256;
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
    int threads = 256;
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
    int threads = 256;
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
    int threads = 256;
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
