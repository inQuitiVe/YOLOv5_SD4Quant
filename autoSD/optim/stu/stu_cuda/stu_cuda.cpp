#include <torch/extension.h>
#include "stu_cuda.h"
#include <tuple>

using namespace at;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// for version < pytorch 1.5
/*
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
*/

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor stu_floatsd4_cuda(Tensor a, Tensor b, int offset);

void fsd_update_cuda(Tensor a, Tensor b, Tensor c, int offset, int mode);

Tensor get_sd_value_cuda(Tensor a, Tensor b, int offset, int mode, int cg);

void init_group_cuda(Tensor a, int mode);
//=================================

Tensor stu_floatsd4(Tensor a, Tensor b, int offset) { //a: master copy / b: update value / c: offset of weights
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  return stu_floatsd4_cuda(a, b, offset);
}

//inplace function for single-trigger-update
void fsd_update(Tensor a, Tensor b, Tensor c, int offset, int mode) { //a: sd group / b: exp / c: update value / offset: weight offset / mode: 0(sd8) 1(sd4)
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  fsd_update_cuda(a, b, c, offset, mode);
}

//convert sd_group & exp to float
Tensor get_sd_value(Tensor a, Tensor b, int offset, int mode, int cg) { //a: sd group / b: exp / offset: weight offset / mode: 0(sd8) 1(sd4) / cg: compute group
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  return get_sd_value_cuda(a, b, offset, mode, cg);
}

//initial sd group
void init_group(Tensor a, int mode) {
  CHECK_INPUT(a);
  init_group_cuda(a, mode);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stu_floatsd4", &stu_floatsd4, "STU for floatsd4 (CUDA)");
  m.def("fsd_update", &fsd_update, "STU for floatsd4 / floatsd8 (CUDA)");
  m.def("get_sd_value", &get_sd_value, "Convert FloatSD format to FP32 (CUDA)");
  m.def("init_group", &init_group, "Random initial SD group (CUDA)");
}
