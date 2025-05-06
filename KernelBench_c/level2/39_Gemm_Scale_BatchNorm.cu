#include <torch/extension.h>
#include <vector>

// Device function for GEMM
at::Tensor gemm(at::Tensor x, at::Tensor weight, at::Tensor bias) {
    return at::addmm(bias, x, weight.t());
}

// Device function for Scaling
__host__ __device__ at::Tensor scale(at::Tensor x, at::Tensor scale_factor) {
    return x * scale_factor;
}

// Device function for Batch Normalization
__host__ __device__ at::Tensor batch_norm(
    at::Tensor x,
    at::Tensor bn_weight,
    at::Tensor bn_bias,
    at::Tensor running_mean,
    at::Tensor running_var,
    double eps,
    double momentum
) {
    return at::batch_norm(
        x,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        /*training=*/true,
        /*momentum=*/momentum,
        /*eps=*/eps,
        /*cudnn_enabled=*/true);
}

// Kernel function
at::Tensor forward(
    at::Tensor x,
    double eps,
    double momentum,
    at::Tensor running_mean,
    at::Tensor running_var,
    at::Tensor gemm_weight,
    at::Tensor gemm_bias,
    at::Tensor scale_factor,
    at::Tensor bn_weight,
    at::Tensor bn_bias
) {
    // Perform GEMM
    x = gemm(x, gemm_weight, gemm_bias);

    // Perform Scaling
    x = scale(x, scale_factor);

    // Perform Batch Normalization
    x = batch_norm(x, bn_weight, bn_bias, running_mean, running_var, eps, momentum);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom module forward function");
}