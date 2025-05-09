void default_function_kernel(float* compute, float* compute_1, float* compute_2, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 9; ++i0) {
    for (int32_t i1 = 0; i1 < 12; ++i1) {
      for (int32_t i2 = 0; i2 < 16; ++i2) {
        compute[(((i0 * 192) + (i1 * 16)) + i2)] = atanhf(ph_0[(((i0 * 192) + (i1 * 16)) + i2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 9; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 12; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 16; ++i2_1) {
        compute_1[(((i0_1 * 192) + (i1_1 * 16)) + i2_1)] = cosf(atanf(ph_0[(((i0_1 * 192) + (i1_1 * 16)) + i2_1)]));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1728; ++i0_i1_fused_i2_fused) {
    compute_2[i0_i1_fused_i2_fused] = atanf(ph_0[i0_i1_fused_i2_fused]);
  }
}

