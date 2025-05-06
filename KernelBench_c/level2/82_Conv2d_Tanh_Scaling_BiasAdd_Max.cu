#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cfloat>

// This fused kernel combines convolution (with bias), tanh activation, scaling, extra bias addition, and max pooling.
// Each thread computes one pooled output element by iterating over its pooling window, computing the convolution on the fly.

template <int KERNEL_H, int KERNEL_W>
__global__ void fused_conv_pool_kernel(
    const float* __restrict__ x,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const float scaling_factor,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int pool_kernel_size,
    const int out_h,    // conv output height = in_h - kernel_h + 1
    const int out_w,    // conv output width  = in_w - kernel_w + 1
    const int pooled_h, // pooled output height = out_h / pool_kernel_size
    const int pooled_w  // pooled output width  = out_w / pool_kernel_size
) {
    // Each thread produces one pooled output value
    int pw = blockIdx.x * blockDim.x + threadIdx.x; // pooled width index
    int ph = blockIdx.y * blockDim.y + threadIdx.y; // pooled height index
    int index_z = blockIdx.z; // combined index for batch and channel
    int n = index_z / out_channels;
    int oc = index_z % out_channels;

    if (n < batch_size && ph < pooled_h && pw < pooled_w) {
        // Compute top-left index in conv output corresponding to this pooling window
        int conv_oh_start = ph * pool_kernel_size;
        int conv_ow_start = pw * pool_kernel_size;

        float max_val = -FLT_MAX;
        // Loop over the pooling window
        for (int py = 0; py < pool_kernel_size; py++) {
            for (int px = 0; px < pool_kernel_size; px++) {
                int conv_oh = conv_oh_start + py;
                int conv_ow = conv_ow_start + px;
                // Safety check, though pooled dims are computed to be valid
                if (conv_oh < out_h && conv_ow < out_w) {
                    float val = conv_bias[oc];
                    // Compute convolution for this conv output element
                    for (int ic = 0; ic < in_channels; ic++) {
                        // Calculate base offsets for input and weight
                        int input_base = ((n * in_channels + ic) * in_h);
                        int weight_base = ((oc * in_channels + ic) * KERNEL_H);
                        #pragma unroll
                        for (int kh = 0; kh < KERNEL_H; kh++) {
                            #pragma unroll
                            for (int kw = 0; kw < KERNEL_W; kw++) {
                                int in_row = conv_oh + kh;
                                int in_col = conv_ow + kw;
                                int input_idx = (input_base + in_row) * in_w + in_col;
                                int weight_idx = (weight_base + kh) * KERNEL_W + kw;
                                val += x[input_idx] * conv_weight[weight_idx];
                            }
                        }
                    }
                    // Apply tanh activation, scaling and add extra bias
                    val = tanhf(val) * scaling_factor + bias[oc];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        // Write the pooled output for this thread
        int out_idx = ((n * out_channels + oc) * pooled_h + ph) * pooled_w + pw;
        out[out_idx] = max_val;
    }
}

// Forward function
// This function sets up the problem dimensions and launches the fused kernel
// It supports 3x3 and 5x5 convolution kernels (the most common cases) with unrolling optimizations

torch::Tensor forward_cuda(
    torch::Tensor x,
    double scaling_factor,
    int pool_kernel_size,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);

    const int out_channels = conv_weight.size(0);
    const int kernel_h = conv_weight.size(2);
    const int kernel_w = conv_weight.size(3);

    // Compute convolution output dimensions (no padding, stride = 1)
    const int out_h = in_h - kernel_h + 1;
    const int out_w = in_w - kernel_w + 1;

    TORCH_CHECK(out_h > 0 && out_w > 0, "Invalid convolution output dimensions");

    // Compute pooled (max pool) output dimensions
    const int pooled_h = out_h / pool_kernel_size;
    const int pooled_w = out_w / pool_kernel_size;
    TORCH_CHECK(pooled_h > 0 && pooled_w > 0, "Invalid pooled output dimensions");

    auto options = x.options();
    auto pooled_out = torch::empty({batch_size, out_channels, pooled_h, pooled_w}, options);

    // Configure grid and block dimensions based on pooled output size
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (pooled_w + blockDim.x - 1) / blockDim.x,
        (pooled_h + blockDim.y - 1) / blockDim.y,
        batch_size * out_channels
    );

    // Launch the templated fused kernel based on kernel size
    if (kernel_h == 3 && kernel_w == 3) {
        fused_conv_pool_kernel<3, 3><<<gridDim, blockDim>>>(
            x.data_ptr<float>(),
            conv_weight.data_ptr<float>(),
            conv_bias.data_ptr<float>(),
            bias.data_ptr<float>(),
            pooled_out.data_ptr<float>(),
            static_cast<float>(scaling_factor),
            batch_size, in_channels, in_h, in_w,
            out_channels,
            pool_kernel_size,
            out_h, out_w,
            pooled_h, pooled_w
        );
    } else if (kernel_h == 5 && kernel_w == 5) {
        fused_conv_pool_kernel<5, 5><<<gridDim, blockDim>>>(
            x.data_ptr<float>(),
            conv_weight.data_ptr<float>(),
            conv_bias.data_ptr<float>(),
            bias.data_ptr<float>(),
            pooled_out.data_ptr<float>(),
            static_cast<float>(scaling_factor),
            batch_size, in_channels, in_h, in_w,
            out_channels,
            pool_kernel_size,
            out_h, out_w,
            pooled_h, pooled_w
        );
    } else {
        TORCH_CHECK(false, "Only 3x3 and 5x5 kernels are supported in the fused version");
    }

    return pooled_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Fused conv-tanh-scale-add and max pool forward (CUDA)");
}
