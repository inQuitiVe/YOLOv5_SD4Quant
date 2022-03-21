#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>
#include <tuple>
#include <ATen/ATen.h>
#include "stu_cuda.h"
#include "stu_kernel.h"
#include <stdio.h>

using namespace at;


Tensor stu_floatsd4_cuda(Tensor a, Tensor b, int offset) {
  //printf("cc");
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  stu_floatsd4_kernel<<<blockNums, blockSize>>>(a.data<int>(),
                                                b.data<float>(),
                                                o.data<int>(),
                                                offset,
                                                size);
  return o;
}

void fsd_update_cuda(Tensor a, Tensor b, Tensor c, int offset, int mode) {
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fsd_update_kernel<<<blockNums, blockSize>>>(a.data<int>(),
                                              b.data<int>(),
                                              c.data<float>(),
                                              offset,
                                              mode,
                                              size);
}


Tensor get_sd_value_cuda(Tensor a, Tensor b, int offset, int mode, int cg) {
  auto o = zeros_like(a, a.options().dtype(kFloat));
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  get_sd_value_kernel<<<blockNums, blockSize>>>(a.data<int>(),
                                                b.data<int>(),
                                                o.data<float>(),
                                                offset,
                                                mode,
                                                cg,
                                                size);
  return o;
}

void init_group_cuda(Tensor a, int mode) {
  auto rand_ints = randint_like(a, INT_MAX);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  init_group_kernel<<<blockNums, blockSize>>>(a.data<int>(),
                                              rand_ints.data<int>(),
                                              mode,
                                              size);
}