#include "cuda_ops.hpp"
#include "cpu_ops.hpp"
#include "examples_helpers.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <format>
#include <span>
#include <cuda_runtime.h>

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
                  const std::string& what = "CUDA", const std::string& to = "CPU", long long ops = -1) {
    std::cout << std::left << std::setw(20) << name 
              << " | Size: " << std::setw(10) << size 
              << std::format(" | {}: ", what) << std::fixed << std::setprecision(4) << std::setw(10) << what_ms << " ms";
    
    if (ops > 0) {
        float gflops = (float)ops / (what_ms * 1e6);
        std::cout << " (" << std::fixed << std::setprecision(2) << std::setw(8) << gflops << " GFLOPS)";
    }

    if (to_ms > 0) {
        std::cout << std::format(" | {}: ", to) << std::fixed << std::setprecision(4) << std::setw(10) << to_ms << " ms";
        if (ops > 0) {
            float gflops_to = (float)ops / (to_ms * 1e6);
            std::cout << " (" << std::fixed << std::setprecision(2) << std::setw(8) << gflops_to << " GFLOPS)";
        }
        std::cout << " | Speedup: " << std::fixed << std::setprecision(2) << to_ms / what_ms << "x";
    }
    std::cout << " | " << (correct ? "PASSED" : "FAILED") << std::endl;
}

void matmul() {
        int sizes[] = {128, 256, 512, 1024, 2048};
        for (int n : sizes) {
            auto A = generate_random_data(n * n);
            auto B = generate_random_data(n * n);
            
            auto gpu_res = cuda::matmul(A, B, n, n, n);
            std::vector<float> cpu_res(n * n);
            cpu::matmul(A.data(), B.data(), cpu_res.data(), n, n, n);
            bool correct = check_correctness(gpu_res, cpu_res);

            float avg_gpu = benchmark([&]() {
                cuda::matmul(A, B, n, n, n);
            }, 5);
            float avg_cpu = benchmark([&]() {
                cpu::matmul(A.data(), B.data(), cpu_res.data(), n, n, n);
            }, 1);
            long long ops = 2LL * n * n * n;
            print_result("Matmul", n, avg_gpu, avg_cpu, correct, "CUDA", "CPU", ops);
        }
    }

void matmul_cpu() {
        int sizes[] = {128, 256, 512};
        for (int n : sizes) {
            auto A = generate_random_data(n * n);
            auto B = generate_random_data(n * n);
            
            std::vector<float> opt_res(n * n);
            std::vector<float> naive_res(n * n);
            cpu::matmul(A.data(), B.data(), opt_res.data(), n, n, n);
            cpu::matmul_naive(A.data(), B.data(), naive_res.data(), n, n, n);
            bool correct = check_correctness(opt_res, naive_res);

            float avg_opt = benchmark([&]() {
                cpu::matmul(A.data(), B.data(), opt_res.data(), n, n, n);
            }, 100);
            float avg_naive = benchmark([&]() {
                cpu::matmul_naive(A.data(), B.data(), naive_res.data(), n, n, n);
            }, 100);
            long long ops = 2LL * n * n * n;
            print_result("Matmul (CPU)", n, avg_opt, avg_naive, correct, "OPT", "NAIVE", ops);
        }
    }

void matmul_opt() {
        int sizes[] = {128, 256, 1024, 2000, 2048, 4096};
        for (int n : sizes) {
            auto A = generate_random_data(n * n);
            auto B = generate_random_data(n * n);
            
            auto opt_res = cuda::matmul(A, B, n, n, n);
            auto naive_res = cuda::matmul_naive(A, B, n, n, n);
            bool correct = check_correctness(opt_res, naive_res);

            float avg_opt = benchmark([&]() {
                cuda::matmul(A, B, n, n, n);
            }, 10);
            float avg_naive = benchmark([&]() {
                cuda::matmul_naive(A, B, n, n, n);
            }, 10);
            long long ops = 2LL * n * n * n;
            print_result("Matmul (CUDA)", n, avg_opt, avg_naive, correct, "OPT", "NAIVE", ops);
        }
    }

void matmul_vs_cublas() {
        int sizes[] = {128, 256, 1024, 2000, 2048, 4096};
        for (int n : sizes) {
            auto A = generate_random_data(n * n);
            auto B = generate_random_data(n * n);
            
            auto opt_res = cuda::matmul(A, B, n, n, n);
            auto cublas_res = cuda::matmul_cublas(A, B, n, n, n);
            bool correct = check_correctness(opt_res, cublas_res, 1e-2);

            float avg_opt = benchmark([&]() {
                cuda::matmul(A, B, n, n, n);
            }, 10);
            float avg_cublas = benchmark([&]() {
                cuda::matmul_cublas(A, B, n, n, n);
            }, 10);
            long long ops = 2LL * n * n * n;
            print_result("Matmul (vs cuBLAS)", n, avg_opt, avg_cublas, correct, "OPT", "cuBLAS", ops);
        }
    }

void matmul_tc() {
    int sizes[] = {1024, 2048, 4096};
    for (int n : sizes) {
        auto A = generate_random_data(n * n);
        auto B = generate_random_data(n * n);
        
        auto wmma_res = cuda::matmul_tc(A, B, n, n, n);
        auto opt_res = cuda::matmul(A, B, n, n, n);
        auto cublas_res = cuda::matmul_cublas(A, B, n, n, n);
        
        bool correct = check_correctness(wmma_res, opt_res, 5e-1);

        float avg_wmma = benchmark([&]() {
            cuda::matmul_tc(A, B, n, n, n);
        }, 10);
        float avg_opt = benchmark([&]() {
            cuda::matmul(A, B, n, n, n);
        }, 10);

        long long ops = 2LL * n * n * n;
        print_result("Matmul WMMA", n, avg_wmma, avg_opt, correct, "WMMA", "OPT", ops);
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
            print_result("Binary Op (+)", n, avg_gpu, avg_cpu, correct, "CUDA", "CPU", (long long)n);
        }
}

void softmax() {
        int N = 1024;
        int C = 256; 
        int size = N * C;
        auto input = generate_random_data(size);
        float cpu_res[size];
        float gpu_res[size];

        cuda::softmax(input.data(), gpu_res, N, C);
        cpu::softmax(input.data(), cpu_res, N, C);
        bool correct = check_correctness(gpu_res, cpu_res, size);

        float avg_gpu = benchmark([&]() {
            cuda::softmax(input.data(), gpu_res, N, C);
        }, 10);
        float avg_cpu = benchmark([&]() {
            cpu::softmax(input.data(), cpu_res, N, C);
        }, 5);
        long long ops = 5LL * N * C;
        print_result("Softmax", N * C, avg_gpu, avg_cpu, correct, "CUDA", "CPU", ops);
}

void reduction() {
        int sizes[] = {1000000, 10000000};
        for (int n : sizes) {
            auto input = generate_random_data(n);
            std::span<const float> span(input);
            
            auto gpu_res = cuda::reduction(ReductionOp::MAX, span);
            auto cpu_res = *std::max_element(input.begin(), input.end());
            bool correct = check_correctness(gpu_res, cpu_res);

            float avg_gpu = benchmark([&]() {
                cuda::reduction(ReductionOp::MAX, span);
            }, 10);
            float avg_cpu = benchmark([&]() {
                *std::max_element(input.begin(), input.end());
            }, 5);
            print_result("Reduction (MAX)", n, avg_gpu, avg_cpu, correct, "CUDA", "CPU", (long long)n);
        }
    }

void im2col() {
    struct Config {
        int B, H, W, C, K, S, P;
    };
    std::vector<Config> configs = {
        {1, 28, 28, 1, 3, 1, 1},
        {1, 64, 64, 3, 3, 1, 1},
        {1, 128, 128, 3, 3, 1, 1},
        {8, 64, 64, 16, 3, 1, 1}
    };

    for (const auto& conf : configs) {
        int size = conf.B * conf.H * conf.W * conf.C;
        auto data = generate_random_data(size);
        
        auto x_opt = Tensor::init({conf.B, conf.H, conf.W, conf.C}, data, Device::CPU);
        auto x_naive = Tensor::init({conf.B, conf.C, conf.H, conf.W}, data, Device::CPU);
        
        float avg_opt = benchmark([&]() {
            (void)x_opt->im2col(conf.K, conf.S, conf.P);
        }, 10);
        
        float avg_naive = benchmark([&]() {
            (void)x_naive->im2col_naive(conf.K, conf.S, conf.P);
        }, 10);
        
        bool correct = true;
        if (conf.C == 1) {
            auto res_opt = x_opt->im2col(conf.K, conf.S, conf.P);
            auto res_naive = x_naive->im2col_naive(conf.K, conf.S, conf.P);
            correct = check_correctness(res_opt->values_vec(), res_naive->values_vec());
        }

        print_result("im2col", size, avg_opt, avg_naive, correct, "OPT", "NAIVE");
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        for (int i = 0; i < argc; ++i) {
            if (strcmp(argv[i], "matmul") == 0) matmul();
            if (strcmp(argv[i], "matmul_cpu") == 0) matmul_cpu();
            if (strcmp(argv[i], "matmul_opt") == 0) matmul_opt();
            if (strcmp(argv[i], "matmul_cublas") == 0) matmul_vs_cublas();
            if (strcmp(argv[i], "matmul_tc") == 0) matmul_tc();
            if (strcmp(argv[i], "binary") == 0) binary();
            if (strcmp(argv[i], "softmax") == 0) softmax();
            if (strcmp(argv[i], "reduction") == 0) reduction();
            if (strcmp(argv[i], "im2col") == 0) im2col();
        }
    } else {
        matmul();
        binary();
        softmax();
        reduction();
        im2col();
    }
    return 0;
}
