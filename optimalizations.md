# GEMM
## CPU
- **Tiling (Blocking):** Implemented tiling for X, Y, and K dimensions (64x32x32) to improve cache locality and minimize cache misses.
- **Multi-threading:** Utilized OpenMP (`#pragma omp parallel for`) with loop collapsing and a fixed thread count (8) to parallelize the computation across CPU cores.
- **SIMD Vectorization (AVX-512):** Leveraged AVX-512 intrinsics to process 16 floating-point elements in a single instruction cycle. AVX-512 fully utilizes cache lines.
- **Fused Multiply-Add (FMA):** Used `_mm512_fmadd_ps` to perform multiplication and addition in a single step, improving performance and numerical precision.
- **Raw Pointer Access:** Used direct pointer access to vector data to avoid the overhead of `std::vector` bounds checking or iterator abstractions in hot loops.

more:
- manual unrolling with full utility of zmm
- aligned memory