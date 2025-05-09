extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ T_divide, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) < 9) {
    T_divide[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = asinhf((ph_0[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] * (ph_0[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] - ph_3[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))])));
}

extern "C" __global__ void __launch_bounds__(2) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = atanhf((ph_0[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] * (ph_0[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] - ph_3[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))])));
}

