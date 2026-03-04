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