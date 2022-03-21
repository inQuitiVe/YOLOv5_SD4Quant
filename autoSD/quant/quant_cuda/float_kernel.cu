#include "quant_kernel.h"
#include "bit_helper.cu"
#include <stdio.h>

//fp8
__global__ void float_kernel_fp8_152_dynamic(float* __restrict__ a,
                                             float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_fp8_152_float_nearest_dynamic(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_fp8_152_clip_dynamic(float* __restrict__ a,
                                                  float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_fp8_152_clip_float_nearest_dynamic(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_fp8_143_dynamic(float* __restrict__ a,
                                             float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_fp8_143_float_nearest_dynamic(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_fp8_143_clip_dynamic(float* __restrict__ a,
                                                  float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_fp8_143_clip_float_nearest_dynamic(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_fp8_134_dynamic(float* __restrict__ a,
                                             float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_fp8_134_float_nearest_dynamic(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_fp8_134_clip_dynamic(float* __restrict__ a,
                                             float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_fp8_134_clip_float_nearest_dynamic(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

//floatsd8
__global__ void float_kernel_floatsd8_dynamic(float* __restrict__ a,
                                              float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {

    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    //unsigned int quantize = round_floatsd8_dynamic_332(old_num, bias);
    //unsigned int quantize = round_floatsd8_dynamic_233(old_num, bias);
    unsigned int quantize = round_floatsd8_dynamic_323(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

//floatsd4
__global__ void float_kernel_floatsd4_dynamic(float* __restrict__ a,
                                              float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_floatsd4_float_dynamic(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_floatsd4_ex_dynamic(float* __restrict__ a,
                                                 float* o, int bias, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_floatsd4_ex_float_dynamic(old_num, bias);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_floatsd4_ex_dynamic_2(float* __restrict__ a,
                                                   float* o, int* __restrict__ bias, int size) {
  int column = blockIdx.x * blockDim.x + threadIdx.x; //same offset
  int index = blockIdx.y * size + column;
  if (column < size) { 
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_floatsd4_ex_float_dynamic(old_num, bias[blockIdx.y]);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

//fp
__global__ void float_kernel_fp_dynamic(float* __restrict__ a,
                                        float* o, int bias, int exp, int mantissa, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_fp_float_nearest_dynamic(old_num, bias, exp, mantissa);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_fp_clip_dynamic(float* __restrict__ a,
                                             float* o, int bias, int exp, int mantissa, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_fp_clip_float_nearest_dynamic(old_num, bias, exp, mantissa);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

__global__ void float_kernel_fp_clip_dynamic_stochastic(float* __restrict__ a,
                                                        float* o, int bias, int exp, int mantissa, 
                                                        int* __restrict__ r, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int quantize = round_fp_clip_float_stochastic_dynamic(old_num, bias, exp, mantissa, rand_prob);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}