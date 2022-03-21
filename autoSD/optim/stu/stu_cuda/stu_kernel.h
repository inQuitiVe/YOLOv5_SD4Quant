#include <stdint.h>
#include <ATen/ATen.h>


__global__ void stu_floatsd4_kernel(int* __restrict__ a,
                                    float* __restrict__ b,
                                    int* o,
                                    int offset,
                                    int size);


__global__ void fsd_update_kernel(int* __restrict__ a,
								  int* __restrict__ b,
                                  float* __restrict__ c,
                                  int offset,
                                  int mode,
                                  int size);

__global__ void get_sd_value_kernel(int* __restrict__ a,
								    int* __restrict__ b,
                                    float* o,
                                  	int offset,
                                  	int mode,
                                  	int cg,
                                  	int size);

__global__ void init_group_kernel(int* __restrict__ a,
								  int* __restrict__ r,
                                  int mode,
                                  int size);