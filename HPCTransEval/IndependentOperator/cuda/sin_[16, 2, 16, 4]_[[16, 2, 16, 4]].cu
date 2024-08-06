extern "C" __global__ void __launch_bounds__(4) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = __sinf(data[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))]);
}

