#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ conv_weight,
    const scalar_t* __restrict__ conv_bias,
    const scalar_t* __restrict__ bias,
    const scalar_t constant_value,
    const scalar_t scaling_factor,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_h,
    const int in_w,
    const int kernel_h,
    const int kernel_w,
    const int out_h,
    const int out_w,
    const int total_elements
) {
    // 2D block configuration
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const int bid = blockIdx.x;
    const int global_idx = bid * block_size + tid;
    
    if (global_idx >= total_elements) return;

    // Decompose global index into tensor dimensions
    const int n = global_idx / (out_channels * out_h * out_w);
    int remainder = global_idx % (out_channels * out_h * out_w);
    const int c_out = remainder / (out_h * out_w);
    remainder = remainder % (out_h * out_w);
    const int h_out = remainder / out_w;
    const int w_out = remainder % out_w;

    scalar_t sum = conv_bias[c_out];

    // Assuming most common case of 3x3 kernel
    if (kernel_h == 3 && kernel_w == 3) {
        #pragma unroll
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            const int x_base = n * in_channels * in_h * in_w + c_in * in_h * in_w;
            const int w_base = c_out * in_channels * 9 + c_in * 9;
            
            // Load weights into registers
            const scalar_t w0 = conv_weight[w_base];
            const scalar_t w1 = conv_weight[w_base + 1];
            const scalar_t w2 = conv_weight[w_base + 2];
            const scalar_t w3 = conv_weight[w_base + 3];
            const scalar_t w4 = conv_weight[w_base + 4];
            const scalar_t w5 = conv_weight[w_base + 5];
            const scalar_t w6 = conv_weight[w_base + 6];
            const scalar_t w7 = conv_weight[w_base + 7];
            const scalar_t w8 = conv_weight[w_base + 8];

            // Load input values
            const scalar_t x00 = x[x_base + (h_out + 0) * in_w + (w_out + 0)];
            const scalar_t x01 = x[x_base + (h_out + 0) * in_w + (w_out + 1)];
            const scalar_t x02 = x[x_base + (h_out + 0) * in_w + (w_out + 2)];
            const scalar_t x10 = x[x_base + (h_out + 1) * in_w + (w_out + 0)];
            const scalar_t x11 = x[x_base + (h_out + 1) * in_w + (w_out + 1)];
            const scalar_t x12 = x[x_base + (h_out + 1) * in_w + (w_out + 2)];
            const scalar_t x20 = x[x_base + (h_out + 2) * in_w + (w_out + 0)];
            const scalar_t x21 = x[x_base + (h_out + 2) * in_w + (w_out + 1)];
            const scalar_t x22 = x[x_base + (h_out + 2) * in_w + (w_out + 2)];

            sum += x00 * w0 + x01 * w1 + x02 * w2 +
                  x10 * w3 + x11 * w4 + x12 * w5 +
                  x20 * w6 + x21 * w7 + x22 * w8;
        }
    } else {
        #pragma unroll 4
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int h_in = h_out + kh;
                    const int w_in = w_out + kw;
                    
                    const int x_idx = n * in_channels * in_h * in_w +
                                    c_in * in_h * in_w +
                                    h_in * in_w +
                                    w_in;
                    
                    const int w_idx = c_out * in_channels * kernel_h * kernel_w +
                                    c_in * kernel_h * kernel_w +
                                    kh * kernel_w +
                                    kw;
                    
                    sum += x[x_idx] * conv_weight[w_idx];
                }
            }
        }
    }

    sum = sum < constant_value ? sum : constant_value;
    sum += bias[c_out];
    sum *= scaling_factor;

    output[global_idx] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    float constant_value,
    float scaling_factor,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be CUDA contiguous tensor");
    TORCH_CHECK(conv_weight.is_cuda() && conv_weight.is_contiguous(), "conv_weight must be CUDA contiguous tensor");
    TORCH_CHECK(conv_bias.is_cuda() && conv_bias.is_contiguous(), "conv_bias must be CUDA contiguous tensor");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bias must be CUDA contiguous tensor");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);

    const int out_channels = conv_weight.size(0);
    const int kernel_h = conv_weight.size(2);
    const int kernel_w = conv_weight.size(3);

    const int out_h = in_h - kernel_h + 1;
    const int out_w = in_w - kernel_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
    const int total_elements = output.numel();

    // Use 2D block configuration (16x16 = 256 threads)
    dim3 threads(16, 16);
    const int blocks = (total_elements + 256 - 1) / 256;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "forward_kernel", ([&] {
        kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            conv_weight.data_ptr<scalar_t>(),
            conv_bias.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            static_cast<scalar_t>(constant_value),
            static_cast<scalar_t>(scaling_factor),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            out_h,
            out_w,
            total_elements
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom fused convolution-min-bias-scale forward");
}