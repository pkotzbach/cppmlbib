#include "linear.hpp"
#include <cstdio>
#include <stdexcept>
#include "linear_forward.h"
#include <cuda_runtime.h>
#include <string>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

Tensor_ptr Linear::forward(Tensor_ptr input)
{
    if (input->shape.size() != 2 && input->shape[1] != in_size) throw std::invalid_argument("Invalid input size");
    return input->matmul(weights) + biases;
    // Tensor_ptr output = Tensor::init({batch_size, out_size}, true);
    // if (input->device.compare("cpu") == 0 && weights->device.compare("cpu") == 0) {
    //     return input->matmul(weights) + biases;
    // }
    // // TODO: streams?
    // else if (input->device.compare("cuda") == 0 && weights->device.compare("cuda") == 0) {
    //     // double *d_input, *d_weights, *d_bias, *d_output;
    //     // std::vector<double> raw_output, raw_input = input->raw_values(), raw_weights = weights->raw_values(), raw_biases = biases->raw_values();
    //     // raw_output->resize(batch_size * out_size);

    //     // // for (auto i: raw_weights) printf("%f ", i);

    //     // size_t input_bytes = batch_size * in_size * sizeof(double);
    //     // size_t weight_bytes = out_size * in_size * sizeof(double);
    //     // size_t bias_bytes = out_size * sizeof(double);
    //     // size_t output_bytes = batch_size * out_size * sizeof(double);

    //     // CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    //     // CUDA_CHECK(cudaMalloc(&d_weights, weight_bytes));
    //     // CUDA_CHECK(cudaMalloc(&d_bias, bias_bytes));
    //     // CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    //     // CUDA_CHECK(cudaMemcpy(d_input, raw_input->data(), input_bytes, cudaMemcpyHostToDevice));
    //     // CUDA_CHECK(cudaMemcpy(d_weights, raw_weights->data(), weight_bytes, cudaMemcpyHostToDevice));
    //     // CUDA_CHECK(cudaMemcpy(d_bias, raw_biases->data(), bias_bytes, cudaMemcpyHostToDevice));

    //     // launch_linear_forward(
    //     //     d_input, d_weights, d_bias, d_output,
    //     //     batch_size, in_size, out_size
    //     // );

    //     // CUDA_CHECK(cudaGetLastError());
    //     // CUDA_CHECK(cudaDeviceSynchronize());

    //     // CUDA_CHECK(cudaMemcpy(raw_output->data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    //     // for (int i = 0; i < batch_size; i++) {
    //     //     for (int j = 0; j < out_size; j++) {
    //     //         printf("output[%d,%d] = %f\n", i, j, raw_output[i * out_size + j]);
    //     //     }
    //     // }

    //     // CUDA_CHECK(cudaFree(d_input));
    //     // CUDA_CHECK(cudaFree(d_weights));
    //     // CUDA_CHECK(cudaFree(d_bias));
    //     // CUDA_CHECK(cudaFree(d_output));
    // }

    // return output;
}

Tensor_ptr Softmax::forward(Tensor_ptr input)
{
    if (input->shape.size() != 2) throw std::invalid_argument("Invalid input size");
    Tensor_ptr exps = input->exp();
    Tensor_ptr sums = exps->sum(1);

    Tensor_ptr result = Tensor::init(input->shape);
    for (int n = 0; n < input->shape[0]; ++n)
        for (int c = 0; c < input->shape[1]; ++c)
            result->at({n, c}) = exps->at({n, c}) / sums->at({n});
    
    return result;
}
