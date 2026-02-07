#pragma once

void launch_linear_forward(
    const double* d_input,
    const double* d_weights,
    const double* d_bias,
    double* d_output,
    int batch_size,
    int in_size,
    int out_size);
