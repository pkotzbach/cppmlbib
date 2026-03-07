#include "cuda_ops.hpp"
#include "cpu_ops.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>
#include <format>
#include <span>

std::vector<float> generate_random_data(size_t size) {
    std::vector<float> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

template<typename Func>
float benchmark(Func func, int iterations = 10) {
    // Warm-up
    func();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> ms = end - start;
    return ms.count() / iterations;
}

bool check_correctness(const float* a, const float* b, int size, float epsilon = 1e-4) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

bool check_correctness(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-4) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

bool check_correctness(float a, float b, float epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

void print_result(const std::string& name, int size, float what_ms, float to_ms = -1.0, bool correct = true, 
                  const std::string& what = "CUDA", const std::string& to = "CPU") {
    std::cout << std::left << std::setw(20) << name 
              << " | Size: " << std::setw(10) << size 
              << std::format(" | {}: ", what) << std::fixed << std::setprecision(4) << std::setw(10) << what_ms << " ms";
    if (to_ms > 0) {
        std::cout << std::format(" | {}: ", to) << std::setw(10) << to_ms << " ms"
                  << " | Speedup: " << std::setprecision(2) << to_ms / what_ms << "x";
    }
    std::cout << " | " << (correct ? "PASSED" : "FAILED") << std::endl;
}

void matmul() {
        int sizes[] = {128, 256, 512, 1024, 2048};
        for (int n : sizes) {
            auto A = generate_random_data(n * n);
            auto B = generate_random_data(n * n);
            
            auto gpu_res = cuda::matmul(A, B, n, n, n);
            auto cpu_res = cpu::matmul(A, B, n, n, n);
            bool correct = check_correctness(gpu_res, cpu_res);

            float avg_gpu = benchmark([&]() {
                cuda::matmul(A, B, n, n, n);
            }, 5);
            float avg_cpu = benchmark([&]() {
                cpu::matmul(A, B, n, n, n);
            }, 1);
            print_result("Matmul", n, avg_gpu, avg_cpu, correct);
        }
    }

void matmul_cpu() {
        int sizes[] = {128, 256, 512};
        for (int n : sizes) {
            auto A = generate_random_data(n * n);
            auto B = generate_random_data(n * n);
            
            auto opt_res = cpu::matmul(A, B, n, n, n);
            auto naive_res = cpu::matmul_naive(A, B, n, n, n);
            bool correct = check_correctness(opt_res, naive_res);

            float avg_opt = benchmark([&]() {
                cpu::matmul(A, B, n, n, n);
            }, 100);
            float avg_naive = benchmark([&]() {
                cpu::matmul_naive(A, B, n, n, n);
            }, 100);
            print_result("Matmul (CPU)", n, avg_opt, avg_naive, correct, "OPT", "NAIVE");
        }
    }

void binary() {
        int sizes[] = {1000000, 10000000};
        for (int n : sizes) {
            auto A = generate_random_data(n);
            auto B = generate_random_data(n);
            
            auto gpu_res = cuda::binary_op('+', A, B, n);
            std::vector<float> cpu_res(n);
            for (int i = 0; i < n; ++i) {
                cpu_res[i] = A[i] + B[i];
            }
            bool correct = check_correctness(gpu_res, cpu_res);

            float avg_gpu = benchmark([&]() {
                cuda::binary_op('+', A, B, n);
            }, 10);
            float avg_cpu = benchmark([&]() {
                for (int i = 0; i < n; ++i) {
                    cpu_res[i] = A[i] + B[i];
                }
            }, 5);
            print_result("Binary Op (+)", n, avg_gpu, avg_cpu, correct);
        }
}

void softmax() {
        int N = 1024;
        int C = 256; 
        int size = N * C;
        auto input = generate_random_data(size);
        float cpu_res[size];
        float gpu_res[size];

        cuda::softmax(input, gpu_res, N, C);
        cpu::softmax(input, cpu_res, N, C);
        bool correct = check_correctness(gpu_res, cpu_res, size);

        float avg_gpu = benchmark([&]() {
            cuda::softmax(input, gpu_res, N, C);
        }, 10);
        float avg_cpu = benchmark([&]() {
            cpu::softmax(input, cpu_res, N, C);
        }, 5);
        print_result("Softmax", N * C, avg_gpu, avg_cpu, correct);
}

void reduction() {
        int sizes[] = {1000000, 10000000};
        for (int n : sizes) {
            auto input = generate_random_data(n);
            std::span<const float> span(input);
            
            auto gpu_res = cuda::reduction(MAX, span);
            auto cpu_res = *std::max_element(input.begin(), input.end());
            bool correct = check_correctness(gpu_res, cpu_res);

            float avg_gpu = benchmark([&]() {
                cuda::reduction(MAX, span);
            }, 10);
            float avg_cpu = benchmark([&]() {
                *std::max_element(input.begin(), input.end());
            }, 5);
            print_result("Reduction (MAX)", n, avg_gpu, avg_cpu, correct);
        }
    }

int main(int argc, char* argv[]) {
    if (argc > 1) {
        for (int i = 0; i < argc; ++i) {
            if (strcmp(argv[i], "matmul") == 0) matmul();
            if (strcmp(argv[i], "matmul_cpu") == 0) matmul_cpu();
            if (strcmp(argv[i], "binary") == 0) binary();
            if (strcmp(argv[i], "softmax") == 0) softmax();
            if (strcmp(argv[i], "reduction") == 0) reduction();
        }
    } else {
        matmul();
        binary();
        softmax();
        reduction();
    }
    return 0;
}
