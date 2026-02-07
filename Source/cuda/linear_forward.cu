#include <cuda_runtime.h>
#include <stdio.h>

__global__ void linear_forward_kernel(
    const double* input,
    const double* weights,
    const double* bias,
    double* output,
    int batch_size,
    int in_size,
    int out_size)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // batch
    int x = blockIdx.x * blockDim.x + threadIdx.x; // output neuron

    if (i < batch_size && x < out_size) {
        double sum = bias[x];
        for (int y = 0; y < in_size; ++y) {
            sum += input[i * in_size + y] * weights[x * in_size + y];
        }
        output[i * out_size + x] = sum;
    }
}

void launch_linear_forward(
    const double* d_input,
    const double* d_weights,
    const double* d_bias,
    double* d_output,
    int batch_size,
    int in_size,
    int out_size)
{
    dim3 block(16, 16);
    dim3 grid(
        (out_size + block.x - 1) / block.x,
        (batch_size + block.y - 1) / block.y
    );

    linear_forward_kernel<<<grid, block>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_size, out_size
    );
}
