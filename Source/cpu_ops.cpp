#include "cpu_ops.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <ranges>
#include <cstdlib>

namespace cpu {
std::vector<float> matmul(const std::vector<float> &A, const std::vector<float> &B, int K, int X, int Y)
{
    return matmul(A.data(), B.data(), K, X, Y);
}

std::vector<float> matmul(const float* A_ptr, const float* B_ptr, int K, int X, int Y)
{
    size_t alignment = 64;
    size_t memory = X * Y * sizeof(float);
    size_t aligned_memory = ((memory + alignment - 1) / alignment) * alignment;
    float* C_ptr = static_cast<float*>(std::aligned_alloc(alignment, aligned_memory));
    if (!C_ptr) throw std::bad_alloc();
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
                        __m512 c_vec = _mm512_load_ps(&C_ptr[y * X + x]);
                        
                        #pragma GCC unroll 4
                        for (int k = tile_k; k < end_k; ++k) {
                            __m512 a_vec = _mm512_set1_ps(A_ptr[y * K + k]);
                            __m512 b_vec = _mm512_load_ps(&B_ptr[k * X + x]);
                            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        }

                        _mm512_store_ps(&C_ptr[y * X + x], c_vec);
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

    std::vector<float> C(C_ptr, C_ptr + X * Y);
    std::free(C_ptr);
    return C;
}

std::vector<float> matmul_naive(const std::vector<float> &A, const std::vector<float> &B, int K, int X, int Y)
{
    std::vector<float> output(X * Y, 0.0);

    for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
            for (int k = 0; k < K; ++k)
                output[y * X + x] += A[y * K + k] * B[k * X + x];

    return output;
}


void softmax(const std::vector<float>& input, float* output, int N, int C) {
    for (int i = 0; i < N; ++i) {
        auto row_view = input | std::views::drop(i * C) | std::views::take(C);
        float max_val = std::ranges::max(row_view);

        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            output[i * C + j] = std::exp(input[i * C + j] - max_val);
            sum += output[i * C + j];
        }

        for (int j = 0; j < C; ++j) {
            output[i * C + j] /= sum;
        }
    }
}

} // namespace cpu
