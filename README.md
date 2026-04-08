# cppmlbib

A lightweight C++20 machine learning library featuring a custom autograd engine and high-performance tensor operations. Includes optimized backends for both CPU (AVX-512) and NVIDIA GPUs (CUDA/Tensor Cores).

## Core Components

- **Autograd Engine**: Computational graph supporting automatic differentiation and backpropagation.
- **Tensors**: Multi-dimensional arrays with support for custom striding, broadcasting, and dual-device (CPU/CUDA) memory management.
- **Layers & Ops**: Includes `Linear`, `Convolution` layers, `Softmax`, `Argmax`, `MSE`.
- **Optimizers**: Basic `SGD` implementation with learning rate scheduling support.

## Performance

The library implements several low-level optimizations to maximize hardware utilization:

- **CPU**: AVX-512 SIMD vectorization, OpenMP multi-threading, and cache-aware GEMM tiling.
- **CUDA**: Custom kernels for element-wise ops and GEMM, with additional support for Tensor Cores (via WMMA) and cuBLAS.

Technical notes and benchmarks can be found in [optimalizations.md](optimalizations.md).

## Getting Started

### Prerequisites
- C++20 compiler (GCC 10+, Clang 10+)
- CMake 3.22+
- CUDA Toolkit 11.0+ (for GPU support)
- CPU support for AVX-512 (ZMM registers)

### Build
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Examples
- **Iris Classification**.
- **Benchmarking** to compare CPU vs GPU performance.

## Testing
Unit tests are implemented using GoogleTest:
```bash
cd build
ctest --output-on-failure
```
