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

Tensor fake_log_quantization_cuda(Tensor a) {
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fake_log_quantization_cuda_kernel<<<blockNums, blockSize>>>(a.data<float>(),
                                                              o.data<float>(),
                                                              size);
  return o;
}