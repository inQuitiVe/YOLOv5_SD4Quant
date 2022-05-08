#include <stdint.h>
#include <ATen/ATen.h>


__global__ void fake_log_quantization_cuda_kernel(float* __restrict__ a,
                                                  float* o, int size);