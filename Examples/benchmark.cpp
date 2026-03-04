#include "cuda_ops.hpp"
#include "cpu_ops.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <span>

std::vector<double> generate_random_data(size_t size) {
    std::vector<double> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

template<typename Func>
double benchmark(Func func, int iterations = 10) {
    // Warm-up
    func();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    return ms.count() / iterations;
}

bool check_correctness(const std::vector<double>& a, const std::vector<double>& b, double epsilon = 1e-4) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

bool check_correctness(double a, double b, double epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

void print_result(const std::string& name, int size, double avg_ms, double cpu_ms = -1.0, bool correct = true) {
    std::cout << std::left << std::setw(20) << name 
              << " | Size: " << std::setw(10) << size 
              << " | CUDA: " << std::fixed << std::setprecision(4) << std::setw(10) << avg_ms << " ms";
    if (cpu_ms > 0) {
        std::cout << " | CPU: " << std::setw(10) << cpu_ms << " ms"
                  << " | Speedup: " << std::setprecision(2) << cpu_ms / avg_ms << "x";
    }
    std::cout << " | " << (correct ? "PASSED" : "FAILED") << std::endl;
}

int main() {
    std::cout << "CUDA vs CPU Ops Benchmark & Verification" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    // Matmul
    // {
    //     int sizes[] = {128, 256, 512};
    //     for (int n : sizes) {
    //         auto A = generate_random_data(n * n);
    //         auto B = generate_random_data(n * n);
            
    //         auto gpu_res = cuda::matmul(A, B, n, n, n);
    //         auto cpu_res = cpu::matmul(A, B, n, n, n);
    //         bool correct = check_correctness(gpu_res, cpu_res);

    //         double avg_gpu = benchmark([&]() {
    //             cuda::matmul(A, B, n, n, n);
    //         }, 5);
    //         double avg_cpu = benchmark([&]() {
    //             cpu::matmul(A, B, n, n, n);
    //         }, 1);
    //         print_result("Matmul", n, avg_gpu, avg_cpu, correct);
    //     }
    // }

    // // Binary ops
    // {
    //     int sizes[] = {1000000, 10000000};
    //     for (int n : sizes) {
    //         auto A = generate_random_data(n);
    //         auto B = generate_random_data(n);
            
    //         auto gpu_res = cuda::binary_op('+', A, B, n);
    //         auto cpu_res = cpu::binary_op('+', A, B, n);
    //         bool correct = check_correctness(gpu_res, cpu_res);

    //         double avg_gpu = benchmark([&]() {
    //             cuda::binary_op('+', A, B, n);
    //         }, 10);
    //         double avg_cpu = benchmark([&]() {
    //             cpu::binary_op('+', A, B, n);
    //         }, 5);
    //         print_result("Binary Op (+)", n, avg_gpu, avg_cpu, correct);
    //     }
    // }

    // 3. Softmax Benchmark
    {
        int N = 1024;
        int C = 256; 
        auto input = generate_random_data(N * C);

        auto gpu_res = cuda::softmax(input, N, C);
        auto cpu_res = cpu::softmax(input, N, C);
        bool correct = check_correctness(gpu_res, cpu_res);

        double avg_gpu = benchmark([&]() {
            cuda::softmax(input, N, C);
        }, 10);
        double avg_cpu = benchmark([&]() {
            cpu::softmax(input, N, C);
        }, 5);
        print_result("Softmax", N * C, avg_gpu, avg_cpu, correct);
    }

    // 4. Reduction Benchmark (MAX)
    {
        int sizes[] = {1000000, 10000000};
        for (int n : sizes) {
            auto input = generate_random_data(n);
            std::span<const double> span(input);
            
            auto gpu_res = cuda::reduction(cuda::MAX, span);
            auto cpu_res = cpu::reduction(cpu::MAX, span);
            bool correct = check_correctness(gpu_res, cpu_res);

            double avg_gpu = benchmark([&]() {
                cuda::reduction(cuda::MAX, span);
            }, 10);
            double avg_cpu = benchmark([&]() {
                cpu::reduction(cpu::MAX, span);
            }, 5);
            print_result("Reduction (MAX)", n, avg_gpu, avg_cpu, correct);
        }
    }

    std::cout << "--------------------------------------------------------" << std::endl;
    return 0;
}
