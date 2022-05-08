#include <torch/torch.h>
#include "quant_cuda.h"
#include <tuple>
#include <vector>
#include <stdio.h>
#include <iostream>

using namespace at;


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// for version < pytorch 1.5
/*
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
*/
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor fake_log_quantization(Tensor a) {
  CHECK_INPUT(a);
  return fake_log_quantization_cuda(a);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_log_quantization", &fake_log_quantization, "Fake log quantization");
}
