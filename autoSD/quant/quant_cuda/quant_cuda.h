#include <ATen/ATen.h>
#include <tuple>
#include <vector>

using namespace at;

//===================================================================
//fp8
Tensor float_quantize_fp8_152_dynamic_cuda(Tensor a, int bias);
Tensor float_quantize_fp8_152_clip_dynamic_cuda(Tensor a, int bias);
Tensor float_quantize_fp8_143_dynamic_cuda(Tensor a, int bias);
Tensor float_quantize_fp8_143_clip_dynamic_cuda(Tensor a, int bias);
Tensor float_quantize_fp8_134_dynamic_cuda(Tensor a, int bias);
Tensor float_quantize_fp8_134_clip_dynamic_cuda(Tensor a, int bias);

//floatsd8
Tensor float_quantize_floatsd8_dynamic_cuda(Tensor a, int bias);

//floatsd4
Tensor float_quantize_floatsd4_dynamic_cuda(Tensor a, int bias);
Tensor float_quantize_floatsd4_ex_dynamic_cuda(Tensor a, int bias);
Tensor float_quantize_floatsd4_ex_cwise_dynamic_cuda(Tensor a, Tensor bias);

//fp
Tensor float_quantize_fp_dynamic_cuda(Tensor a, int bias, int exp, int mantissa);
Tensor float_quantize_fp_clip_dynamic_cuda(Tensor a, int bias, int exp, int mantissa);
Tensor float_quantize_fp_clip_cwise_dynamic_cuda(Tensor a, std::vector<int> bias, std::vector<int> exp, std::vector<int> mantissa);
Tensor float_quantize_fp_clip_cwise_dynamic_stochastic_cuda(Tensor a, std::vector<int> bias, std::vector<int> exp, std::vector<int> mantissa);