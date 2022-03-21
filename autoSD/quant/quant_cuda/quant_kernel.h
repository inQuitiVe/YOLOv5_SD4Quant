#include <stdint.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>

//fp8
__global__ void float_kernel_fp8_152_dynamic(float* __restrict__ a,
                                             float* o, int bias, int size);

__global__ void float_kernel_fp8_152_clip_dynamic(float* __restrict__ a,
                                                  float* o, int bias, int size);

__global__ void float_kernel_fp8_143_dynamic(float* __restrict__ a,
                                             float* o, int bias, int size);
                                             
__global__ void float_kernel_fp8_143_clip_dynamic(float* __restrict__ a,
                                                  float* o, int bias, int size);

__global__ void float_kernel_fp8_134_dynamic(float* __restrict__ a,
                                             float* o, int bias, int size);

__global__ void float_kernel_fp8_134_clip_dynamic(float* __restrict__ a,
                                             float* o, int bias, int size);

//floatsd8
__global__ void float_kernel_floatsd8_dynamic(float* __restrict__ a,
                                              float* o, int bias, int size);

//floatsd4
__global__ void float_kernel_floatsd4_dynamic(float* __restrict__ a,
                                              float* o, int bias, int size);

__global__ void float_kernel_floatsd4_ex_dynamic(float* __restrict__ a,
                                                 float* o, int bias, int size);

__global__ void float_kernel_floatsd4_ex_dynamic_2(float* __restrict__ a,
                                                   float* o, int* __restrict__ bias, int size);

//fp
__global__ void float_kernel_fp_dynamic(float* __restrict__ a,
                                        float* o, int bias, int exp, int mantissa, int size);

__global__ void float_kernel_fp_clip_dynamic(float* __restrict__ a,
                                             float* o, int bias, int exp, int mantissa, int size);

__global__ void float_kernel_fp_clip_dynamic_stochastic(float* __restrict__ a,
                                                        float* o, int bias, int exp, int mantissa, 
                                                        int* __restrict__ r, int size);