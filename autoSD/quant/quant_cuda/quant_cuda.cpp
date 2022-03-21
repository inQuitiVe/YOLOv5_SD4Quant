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

//fp8
Tensor float_quantize_fp8_152_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_fp8_152_dynamic_cuda(a, bias);
}

Tensor float_quantize_fp8_152_clip_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_fp8_152_clip_dynamic_cuda(a, bias);
}

Tensor float_quantize_fp8_143_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_fp8_143_dynamic_cuda(a, bias);
}

Tensor float_quantize_fp8_143_clip_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_fp8_143_clip_dynamic_cuda(a, bias);
}

Tensor float_quantize_fp8_134_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_fp8_134_dynamic_cuda(a, bias);
}

Tensor float_quantize_fp8_134_clip_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_fp8_134_clip_dynamic_cuda(a, bias);
}

//floatsd8
Tensor float_quantize_floatsd8_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_floatsd8_dynamic_cuda(a, bias);
}

//floatsd4
Tensor float_quantize_floatsd4_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_floatsd4_dynamic_cuda(a, bias);
}

Tensor float_quantize_floatsd4_ex_dynamic(Tensor a, int bias) {
  CHECK_INPUT(a);
  return float_quantize_floatsd4_ex_dynamic_cuda(a, bias);
}

Tensor float_quantize_floatsd4_ex_cwise_dynamic(Tensor a, Tensor bias) {
  CHECK_INPUT(a);
  CHECK_INPUT(bias);
  return float_quantize_floatsd4_ex_cwise_dynamic_cuda(a, bias);
}

//fp dynamic
Tensor float_quantize_fp_dynamic(Tensor a, int bias, int exp, int mantissa) {
  CHECK_INPUT(a);
  return float_quantize_fp_dynamic_cuda(a, bias, exp, mantissa);
}

Tensor float_quantize_fp_clip_dynamic(Tensor a, int bias, int exp, int mantissa) {
  CHECK_INPUT(a);
  return float_quantize_fp_clip_dynamic_cuda(a, bias, exp, mantissa);
}

Tensor float_quantize_fp_clip_cwise_dynamic(Tensor a, std::vector<int> bias, std::vector<int> exp, std::vector<int> mantissa) {
  CHECK_INPUT(a);
  return float_quantize_fp_clip_cwise_dynamic_cuda(a, bias, exp, mantissa);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //fp8
  m.def("float_quantize_fp8_152_nearest_dynamic", &float_quantize_fp8_152_dynamic, "FP8 152 Quantization dynamic(CUDA)");
  m.def("float_quantize_fp8_152_clip_nearest_dynamic", &float_quantize_fp8_152_clip_dynamic, "FP8 clipped 152 Quantization dynamic(CUDA)");
  m.def("float_quantize_fp8_143_nearest_dynamic", &float_quantize_fp8_143_dynamic, "FP8 143 Quantization dynamic(CUDA)");
  m.def("float_quantize_fp8_143_clip_nearest_dynamic", &float_quantize_fp8_143_clip_dynamic, "FP8 clipped 143 Quantization dynamic(CUDA)");
  m.def("float_quantize_fp8_134_nearest_dynamic", &float_quantize_fp8_134_dynamic, "FP8 134 Quantization dynamic(CUDA)");
  m.def("float_quantize_fp8_134_clip_nearest_dynamic", &float_quantize_fp8_134_clip_dynamic, "FP8 clipped 134 Quantization dynamic(CUDA)");

  //floatsd8
  m.def("float_quantize_floatsd8_dynamic", &float_quantize_floatsd8_dynamic, "FloatSD8 Quantization dynamic(CUDA)");   

  //floatsd4
  m.def("float_quantize_floatsd4_dynamic", &float_quantize_floatsd4_dynamic, "FloatSD4 Quantization dynamic(CUDA)");
  m.def("float_quantize_floatsd4_ex_dynamic", &float_quantize_floatsd4_ex_dynamic, "FloatSD4_ex Quantization dynamic(CUDA)");
  m.def("float_quantize_floatsd4_ex_cwise_dynamic", &float_quantize_floatsd4_ex_cwise_dynamic, "Channel-wise FloatSD4_ex Quantization dynamic(CUDA)");

  //fp
  m.def("float_quantize_fp_dynamic", &float_quantize_fp_dynamic, "FP Quantization dynamic(CUDA)");
  m.def("float_quantize_fp_clip_dynamic", &float_quantize_fp_clip_dynamic, "FP clipped Quantization dynamic(CUDA)");
  m.def("float_quantize_fp_clip_cwise_dynamic", &float_quantize_fp_clip_cwise_dynamic, "Chennel-wise FP clipped Quantization dynamic(CUDA)");
}
