// __global__ void softmax_kernel(const float* input, float* output, int N, int C) {
//     extern __shared__ float sharedArray[];
//     int n = blockIdx.x * blockDim.x + threadIdx.x;
//     int n_abs = threadIdx.x;
//     int c = threadIdx.y;
//     float* exps = (float*)sharedArray;
//     float* exps_sum = (float*)&exps[blockDim.x * C];
    
//     if (n >= N) return;

//     exps[n_abs * C + c] = exp(input[n * C + c]);
    
//     __syncthreads();

//     // one thread per row
//     if (c == 0) {
//         exps_sum[n_abs] = 0.0;
//         for (int i = 0; i < C; ++i) {
//             exps_sum[n_abs] += exps[n_abs * C + i];
//         }
//     }

//     __syncthreads();

//     output[n * C + c] = exps[n_abs * C + c] / exps_sum[n_abs];
// }

// void launch_softmax(const float* input, float* output, int N, int C) {
// #ifdef CUDA_TEST
//     g_cuda_kernel_launches++;
// #endif

//     dim3 block(32, C);
//     size_t shared_memory_bytes = (block.x * C + block.x) * sizeof(float);
//     int grid = cuda::ceil_div(N, block.x);

//     softmax_kernel<<<grid, block, shared_memory_bytes>>>(input, output, N, C);
// }


// ------------------------ SHARED MEMORY KERNEL
// #define BLOCK_DIM 16
// __global__ void matmul_kernel(const float* A, const float* B, float* C,
//                               int K, int X, int Y)
// {
//     extern __shared__ float shared_A[];
//     float* shared_B = shared_A + BLOCK_DIM * BLOCK_DIM;

//     int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
//     int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

//     float sum = 0.0f;

//     for (int i = 0; i < K; i += BLOCK_DIM) {

//         shared_A[threadIdx.y * BLOCK_DIM + threadIdx.x] = A[y * K + i + threadIdx.x];
//         shared_B[threadIdx.y * BLOCK_DIM + threadIdx.x] = B[(i + threadIdx.y) * X + x];

//         __syncthreads();

//         for (int k = 0; k < BLOCK_DIM; ++k) {
//             sum += shared_A[threadIdx.y * BLOCK_DIM + k] *
//                    shared_B[k * BLOCK_DIM + threadIdx.x];
//         }

//         __syncthreads();
//     }

//     if (x < X && y < Y)
//         C[y * X + x] = sum;
// }

// void launch_matmul(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y)
// {
// #ifdef CUDA_TEST
//     g_cuda_kernel_launches++;
// #endif
//     dim3 block(BLOCK_DIM, BLOCK_DIM);
//     dim3 grid(cuda::ceil_div(X, block.x), cuda::ceil_div(Y, block.y));

//     int shared_memory = 2 * sizeof(float) * BLOCK_DIM * BLOCK_DIM; 

//     matmul_kernel<<<grid, block, shared_memory>>>(d_A, d_B, d_C, K, X, Y);
// }


// ---------------- 1D BLOCKING KERNEL

// #define BLOCK_K 8
// #define PER_THREAD 8
// #define BLOCK_X 64
// #define BLOCK_Y 64
// __global__ void matmul_kernel(const float* A, const float* B, float* C, int K, int X, int Y)
// {
//     extern __shared__ float shared_A[];
//     float* shared_B = shared_A + BLOCK_K * BLOCK_Y;

//     int shared_A_x = threadIdx.x % BLOCK_K;
//     int shared_A_y = threadIdx.x / BLOCK_K;
    
//     int shared_B_x = threadIdx.x % BLOCK_X;
//     int shared_B_y = threadIdx.x / BLOCK_X;

//     int thread_x = threadIdx.x % BLOCK_X;
//     int thread_y = threadIdx.x / BLOCK_X;

//     int A_x = threadIdx.x % BLOCK_K;
//     int A_y = blockIdx.y * BLOCK_Y + threadIdx.x / BLOCK_K;

//     int B_x = blockIdx.x * BLOCK_X + threadIdx.x % BLOCK_X;
//     int B_y = threadIdx.x / BLOCK_X;

//     const int sums = PER_THREAD;
//     float sum[sums] = {0};
//     for (int i = 0; i < K; i += BLOCK_K) {
//         A_x = i + shared_A_x;
//         B_y = i + shared_B_y;

//         shared_A[shared_A_y * BLOCK_K + shared_A_x] = A[A_y * K + A_x];
//         shared_B[shared_B_y * BLOCK_X + shared_B_x] = B[B_y * X + B_x];

//         __syncthreads();

//         for (int k = 0; k < BLOCK_K; ++k) {
//             float blocked_B = shared_B[k * BLOCK_X + shared_B_x];
//             for (int y = 0; y < PER_THREAD; ++y) {
//                 int current_y = thread_y * PER_THREAD + y;
//                 sum[y] += shared_A[current_y * BLOCK_K + k] * blocked_B;
//             }
//         }

//         __syncthreads();
//     }

//     for (int i = 0; i < PER_THREAD; ++i) {
//         int C_x = blockIdx.x * BLOCK_X + thread_x;
//         int C_y = blockIdx.y * BLOCK_Y + thread_y * PER_THREAD + i;

//         C[C_y * X + C_x] = sum[i];
//     }
// }

// void launch_matmul(const float* d_A, const float* d_B, float* d_C, int K, int X, int Y)
// {
// #ifdef CUDA_TEST
//     g_cuda_kernel_launches++;
// #endif
//     static_assert(BLOCK_X * BLOCK_K == BLOCK_Y * BLOCK_K);
//     static_assert(BLOCK_Y / PER_THREAD == PER_THREAD);
//     dim3 grid(cuda::ceil_div(X, BLOCK_X), cuda::ceil_div(Y, BLOCK_Y));
//     dim3 block(BLOCK_X * BLOCK_K); // or BLOCK_Y * BLOCK_K

//     int shared_memory = sizeof(float) * (BLOCK_X * BLOCK_K + BLOCK_Y * BLOCK_K); 

//     matmul_kernel<<<grid, block, shared_memory>>>(d_A, d_B, d_C, K, X, Y);
// }