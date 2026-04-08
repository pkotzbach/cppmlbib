#include <cuda_runtime.h>
#include <cuda/cmath>
#include <mma.h>
#include <cuda_fp16.h>
#include <array>

#include "globals.hpp"
#include "cuda_kernels.h"

#ifdef CUDA_TEST
#include "cuda_debug.h"
int g_cuda_matmul_launches = 0;
#endif

constexpr int PER_THREAD = 8;
constexpr int BLOCK_K = 8;
constexpr int BLOCK_X = 64;
constexpr int BLOCK_Y = 64;

__global__ void matmul_kernel_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int K, int X, int Y) {
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

void launch_matmul_naive(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int K, int X, int Y)
{
#ifdef CUDA_TEST
    g_cuda_matmul_launches++;
#endif

    dim3 block(16, 16);
    dim3 grid(cuda::ceil_div(X, block.x), cuda::ceil_div(Y, block.y));

    matmul_kernel_naive<<<grid, block>>>(d_A, d_B, d_C, K, X, Y);
}


__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int K, int X, int Y) {
    __shared__ __align__(16) float sA[BLOCK_Y * BLOCK_K];
    __shared__ __align__(16) float sB[BLOCK_K * BLOCK_X];

    constexpr int block_steps = BLOCK_X / PER_THREAD; // or BLOCK_Y
    
    const int thread_x = threadIdx.x % block_steps;
    const int thread_y = threadIdx.x / block_steps;

    constexpr int VEC_SIZE = 4; // float4
    constexpr int NUM_THREADS = (BLOCK_X / PER_THREAD) * (BLOCK_Y / PER_THREAD);

    constexpr int SA_STEPS = (BLOCK_Y * BLOCK_K) / (VEC_SIZE * NUM_THREADS);
    constexpr int SA_VECS_PER_ROW = BLOCK_K / VEC_SIZE;
    constexpr int SA_ROWS_PER_STEP = NUM_THREADS / SA_VECS_PER_ROW;

    constexpr int SB_STEPS = (BLOCK_K * BLOCK_X) / (VEC_SIZE * NUM_THREADS);
    constexpr int SB_VECS_PER_ROW = BLOCK_X / VEC_SIZE;
    constexpr int SB_ROWS_PER_STEP = NUM_THREADS / SB_VECS_PER_ROW;

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
            int global_a_row = blockIdx.y * BLOCK_Y + a_r;
            int global_a_col = k + a_col;

            if (global_a_row < Y && global_a_col < K) {
                // sA is transposed
                reinterpret_cast<float4*>(&sA[a_r * BLOCK_K + a_col])[0] = reinterpret_cast<const float4*>(&A[global_a_row * K + global_a_col])[0];
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
        // TODO: vectorize
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

void launch_matmul(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int K, int X, int Y)
{
#ifdef CUDA_TEST
    g_cuda_matmul_launches++;
#endif
    static_assert(BLOCK_X == BLOCK_Y);
    static_assert(BLOCK_Y / PER_THREAD == PER_THREAD);
    dim3 grid(cuda::ceil_div(X, BLOCK_X), cuda::ceil_div(Y, BLOCK_Y));
    dim3 block(BLOCK_X / PER_THREAD * BLOCK_Y / PER_THREAD);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, K, X, Y);
}

// wont work for other consts
__global__ void matmul_nonvec_kernel(const float* A, const float* B, float* C, int K, int X, int Y)
{
    __shared__ float sA[BLOCK_Y * BLOCK_K];
    __shared__ float sB[BLOCK_K * BLOCK_X];

    const int block_steps = BLOCK_X / PER_THREAD; // or BLOCK_Y (assuming they are equal)
    
    const int thread_x = threadIdx.x % BLOCK_K;
    const int thread_y = threadIdx.x / BLOCK_K;

    float sum[PER_THREAD * PER_THREAD] = {0};

    float regA[PER_THREAD] = {0};
    float regB[PER_THREAD] = {0};

    for (int k = 0; k < K; k += BLOCK_K) {

        for (int i = 0; i < block_steps; ++i) {
            // square per thread block
            int A_offset = i * BLOCK_K;
            int A_row = blockIdx.y * BLOCK_Y + thread_y + A_offset;
            int A_col = k + thread_x;
            int sA_idx = (thread_y + A_offset) * BLOCK_K + thread_x;
            
            if (A_row < Y && A_col < K)
                sA[sA_idx] = A[A_row * K + A_col];
            else
                sA[sA_idx] = 0;

            // row per thread block
            int B_row = k + i;
            int B_col = blockIdx.x * BLOCK_X + threadIdx.x; // assuming threads in a thread block is equal to BLOCK_X
            int sB_idx = i * BLOCK_X + threadIdx.x;
            
            if (B_row < K && B_col < X)
                sB[sB_idx] = B[B_row * X + B_col];
            else
                sB[sB_idx] = 0;
        }

        __syncthreads();

        // inner loop
        for (int k_in = 0; k_in < BLOCK_K; ++k_in) {
            for (int i = 0; i < PER_THREAD; ++i) {
                regA[i] = sA[(thread_y * PER_THREAD + i) * BLOCK_K + k_in];
                regB[i] = sB[k_in * BLOCK_X + thread_x * PER_THREAD + i];
            }

            for (int y = 0; y < block_steps; ++y) {
                    for (int x = 0; x < block_steps; ++x) {
                        sum[y * PER_THREAD + x] += regA[y] * regB[x];
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

void launch_matmul_nonvec(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y)
{
#ifdef CUDA_TEST
    g_cuda_matmul_launches++;
#endif
    static_assert(BLOCK_X == BLOCK_Y);
    static_assert(BLOCK_Y / PER_THREAD == PER_THREAD);
    dim3 grid(cuda::ceil_div(X, BLOCK_X), cuda::ceil_div(Y, BLOCK_Y));
    dim3 block(BLOCK_X / PER_THREAD * BLOCK_Y / PER_THREAD);

    matmul_nonvec_kernel<<<grid, block>>>(d_A, d_B, d_C, K, X, Y);
}

constexpr int WMMA_Y = 16;
constexpr int WMMA_X = 16;
constexpr int WMMA_K = 16;

__global__ void convert_fp32_to_fp16(const float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = __float2half(in[idx]);
}

__global__ void matmul_tc_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int K, int X, int Y) {
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

void launch_matmul_tc(const float* __restrict__ d_A, const float* __restrict__ d_B, float* __restrict__ d_C, int K, int X, int Y) {
    if (K % 16 != 0 || X % 16 != 0 || Y % 16 != 0) {
        launch_matmul(d_A, d_B, d_C, K, X, Y);
        return;
    }

    half *d_A_half, *d_B_half;
    cudaMalloc(&d_A_half, Y * K * sizeof(half));
    cudaMalloc(&d_B_half, K * X * sizeof(half));

    constexpr int threads = 256;
    convert_fp32_to_fp16<<<cuda::ceil_div(Y * K, threads), threads>>>(d_A, d_A_half, Y * K);
    convert_fp32_to_fp16<<<cuda::ceil_div(K * X, threads), threads>>>(d_B, d_B_half, K * X);

    dim3 block(128); 
    dim3 grid(cuda::ceil_div(X, 32), cuda::ceil_div(Y, 32));

    matmul_tc_kernel<<<grid, block>>>(d_A_half, d_B_half, d_C, K, X, Y);

    cudaFree(d_A_half);
    cudaFree(d_B_half);
}
