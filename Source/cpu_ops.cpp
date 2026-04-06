#include "cpu_ops.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <ranges>
#include <cstdlib>

namespace cpu {
void matmul(const float* A_ptr, const float* B_ptr, float* C_ptr, int K, int X, int Y)
{
    std::fill_n(C_ptr, X * Y, 0.0f);

    constexpr int tile_size_X = 64;
    constexpr int tile_size_Y = 32;
    constexpr int tile_size_K = 32;
    
    constexpr int simd_step = 16; // AVX-512

    // #pragma omp parallel for shared(A_ptr, B_ptr, C_ptr) collapse(2)
    for (int tile_y = 0; tile_y < Y; tile_y += tile_size_Y) {
        for (int tile_x = 0; tile_x < X; tile_x += tile_size_X) {
            int end_y = std::min(tile_y + tile_size_Y, Y);
            int end_x = std::min(tile_x + tile_size_X, X);
            
            for (int tile_k = 0; tile_k < K; tile_k += tile_size_K) {
                int end_k = std::min(tile_k + tile_size_K, K);

                 for (int y = tile_y; y < end_y; ++y) {
                    
                    int x = tile_x;

                    #pragma GCC unroll 4
                    for (; x <= end_x - simd_step; x += simd_step) {
                        __m512 c_vec = _mm512_loadu_ps(&C_ptr[y * X + x]);
                        
                        #pragma GCC unroll 4
                        for (int k = tile_k; k < end_k; ++k) {
                            __m512 a_vec = _mm512_set1_ps(A_ptr[y * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B_ptr[k * X + x]);
                            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        }

                        _mm512_storeu_ps(&C_ptr[y * X + x], c_vec);
                    }
                    
                    // rest
                    for (; x < end_x; ++x) {
                        float sum = C_ptr[y * X + x];
                        for (int k = tile_k; k < end_k; ++k) {
                            sum += A_ptr[y * K + k] * B_ptr[k * X + x];
                        }
                        C_ptr[y * X + x] = sum;
                    }
                }
            }
        }
    }
}

// transpose B and matmul
// [M, K] @ [N, K]^T
void BT_matmul(const float* A_ptr, const float* B_ptr, float* C_ptr, int K, int M, int N)
{
    std::fill_n(C_ptr, M * N, 0.0f);

    constexpr int tile_size_M = 32;
    constexpr int tile_size_N = 64;
    constexpr int tile_size_K = 32;
    
    constexpr int simd_step = 16; // AVX-512

    // #pragma omp parallel for shared(A_ptr, B_ptr, C_ptr) collapse(2)
    for (int start_m = 0; start_m < M; start_m += tile_size_M) {
        for (int start_n = 0; start_n < N; start_n += tile_size_N) {
            int end_M = std::min(start_m + tile_size_M, M);
            int end_N = std::min(start_n + tile_size_N, N);

            for (int start_k = 0; start_k < K; start_k += tile_size_K) {
                int end_k = std::min(start_k + tile_size_K, K);

                alignas(64) float B_tile[tile_size_K][tile_size_N] = {0.0f};
                
                for (int n = start_n; n < end_N; ++n) {
                    for (int k = start_k; k < end_k; ++k) {
                        B_tile[k - start_k][n - start_n] = B_ptr[n * K + k];
                    }
                }

                for (int m = start_m; m < end_M; ++m) {
                    int n = start_n;

                    #pragma GCC unroll 4
                    for (; n <= end_N - simd_step; n += simd_step) {
                        __m512 c_vec = _mm512_loadu_ps(&C_ptr[m * N + n]);
                        
                        for (int k = start_k; k < end_k; ++k) {
                            __m512 a_vec = _mm512_set1_ps(A_ptr[m * K + k]);
                            __m512 b_vec = _mm512_load_ps(&B_tile[k - start_k][n - start_n]);
                            
                            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        }

                        _mm512_storeu_ps(&C_ptr[m * N + n], c_vec);
                    }
                    
                    for (; n < end_N; ++n) {
                        float sum = C_ptr[m * N + n];
                        for (int k = start_k; k < end_k; ++k) {
                            sum += A_ptr[m * K + k] * B_tile[k - start_k][n - start_n];
                        }
                        C_ptr[m * N + n] = sum;
                    }
                }
            }
        }
    }
}

// C = A^T @ B
void AT_matmul(const float* A_ptr, const float* B_ptr, float* C_ptr, int K, int M, int N)
{
    std::fill_n(C_ptr, M * N, 0.0f);

    constexpr int tile_size_M = 32;
    constexpr int tile_size_N = 64;
    constexpr int tile_size_K = 32;
    
    constexpr int simd_step = 16; // AVX-512

    // #pragma omp parallel for shared(A_ptr, B_ptr, C_ptr) collapse(2)
    for (int start_m = 0; start_m < M; start_m += tile_size_M) {
        for (int start_n = 0; start_n < N; start_n += tile_size_N) {
            int end_m = std::min(start_m + tile_size_M, M);
            int end_n = std::min(start_n + tile_size_N, N);

            for (int start_k = 0; start_k < K; start_k += tile_size_K) {
                int end_k = std::min(start_k + tile_size_K, K);

                 for (int m = start_m; m < end_m; ++m) {
                    int n = start_n;

                    #pragma GCC unroll 4
                    for (; n <= end_n - simd_step; n += simd_step) {
                        __m512 c_vec = _mm512_loadu_ps(&C_ptr[m * N + n]);
                        
                        #pragma GCC unroll 4
                        for (int k = start_k; k < end_k; ++k) {
                            __m512 a_vec = _mm512_set1_ps(A_ptr[k * M + m]);
                            __m512 b_vec = _mm512_loadu_ps(&B_ptr[k * N + n]);
                            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        }

                        _mm512_storeu_ps(&C_ptr[m * N + n], c_vec);
                    }
                    
                    // rest
                    for (; n < end_n; ++n) {
                        float sum = C_ptr[m * N + n];
                        for (int k = start_k; k < end_k; ++k) {
                            sum += A_ptr[k * M + m] * B_ptr[k * N + n];
                        }
                        C_ptr[m * N + n] = sum;
                    }
                }
            }
        }
    }
}

void matmul_naive(const float* A_ptr, const float* B_ptr, float* C_ptr, int K, int X, int Y)
{
    std::fill_n(C_ptr, X * Y, 0.0f);

    for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
            for (int k = 0; k < K; ++k)
                C_ptr[y * X + x] += A_ptr[y * K + k] * B_ptr[k * X + x];
}


void softmax(const float* __restrict__ input, float* __restrict__ output, int N, int C)
{
    for (int i = 0; i < N; ++i) {
        const float* __restrict__ in_row  = input  + i * C;
        float* __restrict__ out_row = output + i * C;

        float max_val = in_row[0];
        for (int j = 1; j < C; ++j) {
            max_val = std::max(max_val, in_row[j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            float val = std::exp(in_row[j] - max_val);
            out_row[j] = val;
            sum += val;
        }

        float inv_sum = 1.0f / sum;
        for (int j = 0; j < C; ++j) {
            out_row[j] *= inv_sum;
        }
    }
}

} // namespace cpu
