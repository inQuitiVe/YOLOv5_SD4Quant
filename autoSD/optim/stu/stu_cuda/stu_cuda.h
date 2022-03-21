#include <ATen/ATen.h>
#include <tuple>

using namespace at;

Tensor stu_floatsd4(Tensor a, Tensor b, int offset);

void fsd_update(Tensor a, Tensor b, Tensor c, int offset, int mode);

Tensor get_sd_value(Tensor a, Tensor b, int offset, int mode, int cg);

void init_group(Tensor a, int mode);