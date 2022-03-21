#include "stu_kernel.h"
#include "stu_helper.cu"
#include <stdio.h>

#define GRP_NUM 8
#define BIT_WIDTH 3 //use 3 bit to represent 1 group

__global__ void stu_floatsd4_kernel(int* __restrict__ a,
                                    float* __restrict__ b,
                                    int* o, 
                                    int offset,
                                    int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("cc");
  if (index < size) {
    unsigned int old_sd_group = INT_TO_BITS(&a[index]);
    unsigned int update_value = FLOAT_TO_BITS(&b[index]);
    unsigned int new_sd_group = update_floatsd4(old_sd_group, update_value, offset);
    float new_sd_group_float = BITS_TO_INT(&new_sd_group);
    o[index] = new_sd_group_float;
  }
}

__global__ void fsd_update_kernel(int* __restrict__ a,
                                  int* __restrict__ b,
                                  float* __restrict__ c,
                                  int offset,
                                  int mode,                             
                                  int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("cc");
  if (index < size) {
    int group_width[8] = {3, 3, 3, 3, 3, 3, 3, 3};
    int exp_size;
    int exp_max_value;
    if (mode == 0) { //floatsd 8
      //group_width = {3, 2, 3, 3, 2, 3, 3, 3};
      group_width[1] = 2;
      group_width[4] = 2;
      exp_size = 3;
      exp_max_value = 7;
    }
    else { //floatsd 4
      exp_size = 2;
      exp_max_value = 3;
    }

    int exp_mask = (1 << exp_size) - 1;
    int index_mask; 
    index_mask = (1 << BIT_WIDTH) - 1; //111

    int MC_exp;
    MC_exp = (b[index]) & exp_mask ;

    int tri; // for trigger log command
    int update_sign;
    float update_value = c[index];
    if (update_value >= 0)
      update_sign = 1;
    else
      update_sign = -1;

    update_value = abs(update_value);

    //sd_group => [LSG, ...., MSG] ([G1 G2 ... G8])
    //sd_group MSG ... LSG . <- decimal point
    if (mode == 0){ //sd8
      if ((update_value) >= exp2((double)(18-offset+MC_exp)) )      tri = 8;
      else if ((update_value) >= exp2((double)(16-offset+MC_exp)) & (update_value) < exp2((double)(18-offset+MC_exp)))  tri=7;                          
      else if ((update_value) >= exp2((double)(13-offset+MC_exp)) & (update_value) < exp2((double)(16-offset+MC_exp)))  tri=6;                         
      else if ((update_value) >= exp2((double)(10-offset+MC_exp)) & (update_value) < exp2((double)(13-offset+MC_exp)))  tri=5;                          
      else if ((update_value) >= exp2((double)(8-offset+MC_exp))  & (update_value) < exp2((double)(10-offset+MC_exp)))  tri=4;
      else if ((update_value) >= exp2((double)(5-offset+MC_exp))  & (update_value) < exp2((double)(8-offset+MC_exp)))   tri=3;                      
      else if ((update_value) >= exp2((double)(2-offset+MC_exp))  & (update_value) < exp2((double)(5-offset+MC_exp)))   tri=2;
      else if ((update_value) >= exp2((double)((-1)-offset+MC_exp)) & (update_value)< exp2((double)(2-offset+MC_exp)))  tri=1;
      else tri=0;
    }
    else { //sd4
      if ((update_value) >= exp2((double)(20-offset+MC_exp))) tri = 8;
      else if ((update_value) >= exp2((double)(17-offset+MC_exp)) & (update_value) < exp2((double)(20-offset+MC_exp)))  tri=7;                          
      else if ((update_value) >= exp2((double)(14-offset+MC_exp)) & (update_value) < exp2((double)(17-offset+MC_exp)))  tri=6;                         
      else if ((update_value) >= exp2((double)(11-offset+MC_exp)) & (update_value) < exp2((double)(14-offset+MC_exp)))  tri=5;                          
      else if ((update_value) >= exp2((double)(8-offset+MC_exp))  & (update_value) < exp2((double)(11-offset+MC_exp)))  tri=4;
      else if ((update_value) >= exp2((double)(5-offset+MC_exp))  & (update_value) < exp2((double)(8-offset+MC_exp)))   tri=3;                      
      else if ((update_value) >= exp2((double)(2-offset+MC_exp))  & (update_value) < exp2((double)(5-offset+MC_exp)))   tri=2;
      else if ((update_value) >= exp2((double)((-1)-offset+MC_exp)) & (update_value)< exp2((double)(2-offset+MC_exp)))  tri=1;
      else tri=0;
    }

    if (update_sign == -1)
      tri = -tri;

    int tri_flag = 1;
    int i_tmp;
    int max_value;
    int new_i;
    int new_tri;
    int tri_abs = abs(tri);

    while (tri_flag) {
        if (tri == 0){
            break;
        }
        if ((tri_abs <= GRP_NUM) & (tri_abs > 0)){
            tri_abs = abs(tri);
            max_value = group_width[GRP_NUM - tri_abs] * 2; //ex: 3 bit per group => index = 0~6 <= 3*2
            int shift = BIT_WIDTH * (GRP_NUM - tri_abs);
            int mask_tmp = index_mask << shift;
            int spare_mask = ~mask_tmp;   //

            i_tmp = a[index] & mask_tmp;
            i_tmp = i_tmp >> shift;

            //printf("tri: %d\n", tri);
            //printf("i_tmp: %d\n", i_tmp);
        
            if (tri > 0) { // plus
                new_i = i_tmp + 1; // up trigger
                tri_flag = 0;
                new_tri = tri;
                if ((i_tmp == max_value) & (tri_abs == GRP_NUM)){ // at top of MSG
                    if (MC_exp == exp_max_value){ //already max
                        break;
                    }
                    // GRP 1~7 toward zero
                    for (int ooo = 1; ooo < GRP_NUM; ooo++) { // 
                        int i_ttmmpp; // for exp handle case
                        int shift_tmp = BIT_WIDTH * (GRP_NUM - abs(ooo));
                        int mask_ttmmpp = index_mask << shift_tmp;
                        int spare_mask_ttmmpp = ~mask_ttmmpp;
                        int idx_zero = group_width[GRP_NUM - abs(ooo)];

                        i_ttmmpp = a[index] & mask_ttmmpp;
                        i_ttmmpp = i_ttmmpp >> shift_tmp; // get index of GRP
                        //toward zero
                        if (i_ttmmpp > idx_zero) i_ttmmpp -= 1;
                        if (i_ttmmpp < idx_zero) i_ttmmpp += 1;

                        a[index] = ((a[index] & spare_mask_ttmmpp) + (i_ttmmpp << shift_tmp)); //inplcae return
                    }
                    b[index] += 1; // exp + 1 inplace return
                    MC_exp += 1;
                    break; // finish the task instead, break immediately.
                } //end of exp modify

                // at top of GRP but not MSG
                if (i_tmp == max_value){ 
                    new_i = 0;
                    new_tri = tri + 1; //trigger next group
                    tri_flag = 1;
                } 
                
                a[index] = ((a[index] & spare_mask) + (new_i << shift));
                tri = new_tri;
            } //end plus
            else { //minus
                new_i = i_tmp - 1;
                tri_flag = 0;
                new_tri = tri;
                // // at bottom of MSG
                if ((i_tmp == 0) & (tri_abs == GRP_NUM)){
                    if (MC_exp == exp_max_value){
                        break; //already max
                    }
                    // GRP 1~7 toward zero
                    for (int ooo=1; ooo < GRP_NUM; ooo++){
                        int i_ttmmpp; // for exp handle case
                        int shift_tmp = BIT_WIDTH * (GRP_NUM - abs(ooo));
                        int mask_ttmmpp = index_mask << shift_tmp;
                        int spare_mask_ttmmpp = ~mask_ttmmpp;
                        int idx_zero = group_width[GRP_NUM - abs(ooo)];

                        i_ttmmpp = a[index] & mask_ttmmpp;
                        i_ttmmpp = i_ttmmpp >> shift_tmp; // get index of GRP
                        //toward zero
                        if (i_ttmmpp > idx_zero) i_ttmmpp -= 1;
                        if (i_ttmmpp < idx_zero) i_ttmmpp += 1;

                        a[index] = ((a[index] & spare_mask_ttmmpp) + (i_ttmmpp << shift_tmp)); //inplcae return
                    }
                    b[index] += 1; // exp + 1 inplace return
                    MC_exp += 1;
                    break; // finish the task instead, break immediately.
                } //end of handle MSB bottom
                // at bottom of GRP but not MSG
                if (i_tmp == 0) {
                    new_i = max_value;
                    new_tri = tri - 1;
                    tri_flag = 1;
                }
                a[index] = ((a[index] & spare_mask) + (new_i << shift));
                tri = new_tri;
            } //end of minus
        } //end of update
    } //end of while
  } //end of index < size
}

__global__ void get_sd_value_kernel(int* __restrict__ a,
                                    int* __restrict__ b,
                                    float* o,
                                    int offset,
                                    int mode,
                                    int cg,                            
                                    int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    int group_width[8] = {3, 3, 3, 3, 3, 3, 3, 3};
    int total_width = 24;
    if (mode == 0) { //floatsd 8
      group_width[1] = 2;
      group_width[4] = 2;
      total_width = 22;
    }

    int sd_value = 0;
    int index_mask = (1 << BIT_WIDTH) - 1;
    for (int gg = 0; gg < cg; gg++){
      int width_now = group_width[gg];
      total_width -= width_now;
      int shift = BIT_WIDTH * gg;
      int mask_tmp = index_mask << shift;
      int i_tmp = a[index] & mask_tmp;
      i_tmp = i_tmp >> shift;
      if (i_tmp > width_now) sd_value += exp2((double)(i_tmp - 1 - width_now)) * exp2((double)(total_width));
      else if (i_tmp < width_now) sd_value -= exp2((double)(width_now - i_tmp - 1)) * exp2((double)(total_width));
    }

    float final = static_cast<float>(sd_value) * static_cast<float>(exp2((double)(b[index] - offset)));
    //printf("%f\n", final);
    o[index] = final;
  }
}

__global__ void init_group_kernel(int* __restrict__ a, //a is zero tensor
                                  int* __restrict__ r,
                                  int mode,                           
                                  int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    int group_width[8] = {3, 3, 3, 3, 3, 3, 3, 3};
    if (mode == 0) { //floatsd 8
      //group_width = {3, 2, 3, 3, 2, 3, 3, 3};
      group_width[1] = 2;
      group_width[4] = 2;
    }

    int max_value;
    int index_mask = (1 << BIT_WIDTH) - 1;
    for (int gg = 0; gg < GRP_NUM; gg++){
      int width_now = group_width[gg];
      max_value = 2 * width_now + 1;
      int shift = BIT_WIDTH * gg;
      int mask_tmp = index_mask << shift;
      int spare_mask = ~mask_tmp;
      int i_tmp = r[index] & mask_tmp;
      i_tmp = i_tmp >> shift;
      i_tmp = i_tmp % max_value;
      a[index] = ((a[index] & spare_mask) + (i_tmp << shift));
    }
  }
}