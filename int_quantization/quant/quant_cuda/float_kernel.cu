#include "quant_kernel.h"
#include "bit_helper.cu"
#include <stdio.h>

__global__ void fake_log_quantization_cuda_kernel(float* __restrict__ a,
                                                  float* o, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_log_quantization(old_num);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}