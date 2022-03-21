#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>
#include <tuple>
#include <ATen/ATen.h>
#include "quant_cuda.h"
#include "quant_kernel.h"
#include <stdio.h>
#include <vector>

using namespace at;

//fp8
Tensor float_quantize_fp8_152_dynamic_cuda(Tensor a, int bias) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_fp8_152_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                         o.data<float>(),
                                                         bias,
                                                         size);
  return o;
}

Tensor float_quantize_fp8_152_clip_dynamic_cuda(Tensor a, int bias) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_fp8_152_clip_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                              o.data<float>(),
                                                              bias,
                                                              size);
  return o;
}

Tensor float_quantize_fp8_143_dynamic_cuda(Tensor a, int bias) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_fp8_143_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                         o.data<float>(),
                                                         bias,
                                                         size);
  return o;
}

Tensor float_quantize_fp8_143_clip_dynamic_cuda(Tensor a, int bias) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_fp8_143_clip_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                              o.data<float>(),
                                                              bias,
                                                              size);
  return o;
}

Tensor float_quantize_fp8_134_dynamic_cuda(Tensor a, int bias) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_fp8_134_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                         o.data<float>(),
                                                         bias,
                                                         size);
  return o;
}

Tensor float_quantize_fp8_134_clip_dynamic_cuda(Tensor a, int bias) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_fp8_134_clip_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                         o.data<float>(),
                                                         bias,
                                                         size);
  return o;
}

//floatsd8
Tensor float_quantize_floatsd8_dynamic_cuda(Tensor a, int bias) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_floatsd8_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                          o.data<float>(),
                                                          bias,
                                                          size);
  return o;
}

//floatsd4
Tensor float_quantize_floatsd4_dynamic_cuda(Tensor a, int bias) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_floatsd4_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                          o.data<float>(),
                                                          bias,
                                                          size);
  return o;
}

Tensor float_quantize_floatsd4_ex_dynamic_cuda(Tensor a, int bias) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_floatsd4_ex_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                             o.data<float>(),
                                                             bias,
                                                             size);
  return o;
}

Tensor float_quantize_floatsd4_ex_cwise_dynamic_cuda(Tensor a, Tensor bias) {
  auto o = zeros_like(a);
  int channel = a.size(0);
  int size = a[0].numel();

  int blockSize = 1024;
  dim3 blockNums((size + blockSize - 1) / blockSize, channel);

  float_kernel_floatsd4_ex_dynamic_2<<<blockNums, blockSize>>>(a.data<float>(),
                                                               o.data<float>(),
                                                               bias.data<int>(),
                                                               size);
  return o;
}

//fp
Tensor float_quantize_fp_dynamic_cuda(Tensor a, int bias, int exp, int mantissa) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_fp_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                    o.data<float>(),
                                                    bias,
                                                    exp,
                                                    mantissa,
                                                    size);
  return o;
}

Tensor float_quantize_fp_clip_dynamic_cuda(Tensor a, int bias, int exp, int mantissa) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_fp_clip_dynamic<<<blockNums, blockSize>>>(a.data<float>(),
                                                         o.data<float>(),
                                                         bias,
                                                         exp,
                                                         mantissa,
                                                         size);
  return o;
}

Tensor float_quantize_fp_clip_cwise_dynamic_cuda(Tensor a, std::vector<int> bias, std::vector<int> exp, std::vector<int> mantissa) {
  auto o = zeros_like(a);
  for (int c = 0; c < a.size(0); c += 1){
    int size = a[c].numel();
    int blockSize = 1024;
    int blockNums = (size + blockSize - 1) / blockSize;

    float_kernel_fp_clip_dynamic<<<blockNums, blockSize>>>(a[c].data<float>(),
                                                           o[c].data<float>(),
                                                           bias[c],
                                                           exp[c],
                                                           mantissa[c],
                                                           size);
  }
  return o;
}

Tensor float_quantize_fp_clip_cwise_dynamic_stochastic_cuda(Tensor a, 
                                                            std::vector<int> bias, 
                                                            std::vector<int> exp, 
                                                            std::vector<int> 
                                                            mantissa) {
  auto o = zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  for (int c = 0; c < a.size(0); c += 1){
    int size = a[c].numel();
    int blockSize = 1024;
    int blockNums = (size + blockSize - 1) / blockSize;

    float_kernel_fp_clip_dynamic_stochastic<<<blockNums, blockSize>>>(a[c].data<float>(),
                                                                      o[c].data<float>(),
                                                                      bias[c],
                                                                      exp[c],
                                                                      mantissa[c],
                                                                      rand_ints[c].data<int>(),
                                                                      size);
  }
  return o;
}