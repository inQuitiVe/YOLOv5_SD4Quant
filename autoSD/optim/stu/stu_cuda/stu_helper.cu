#define FLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float*>(x))
#define INT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define BITS_TO_INT(x) (*reinterpret_cast<int*>(x))

//sd_group = 2 bit exp + 8 * 3 bit group
__device__ __forceinline__ unsigned int update_floatsd4(unsigned int sd_group, unsigned int update_value, int offset) {
  //decode sd_group
  unsigned int exp_weight = sd_group / 5764801; // sd_group // 7^8
  unsigned int group_value = sd_group % 5764801;
  unsigned int new_sd_group = 0;
  unsigned int new_group_value = 0;
  //decode update value
  unsigned int sign = update_value >> 31;
  unsigned int exp_up = (update_value >> 23) & 0x00ff;
  //locate leading one
  unsigned int exp_total = exp_weight + offset + 127;

  //update value too big
  if ((exp_up > exp_total) || (exp_up == exp_total)){
    return sd_group;
  }
  else {
    unsigned int exp_diff = exp_total - exp_up - 1;
    if (exp_diff > 23) { //update value too small
      return sd_group;
    }
    else {
      unsigned int group2update = 7 - (exp_diff / 3); //0~7
      unsigned int trigger_value = 1;
      for (int i=0; i<group2update; i++) {
        trigger_value = trigger_value * 7;
      }
      if (sign == 1){ //minus
        if (trigger_value > group_value){
          new_group_value = 0;
        }
        else {
          new_group_value = group_value - trigger_value;
        }
      }
      else { //add
        new_group_value = group_value + trigger_value;
        if (new_group_value > 5764800) { //need exp modification
          if (exp_weight == 3) { //already max => no update
            return sd_group;
          }
          else {
            //exp_weight = exp_weight + 1; TODO
            return sd_group;
          }
        }
      }
    }
  }

  new_sd_group = exp_weight * 5764801 + new_group_value;
  return new_sd_group;
}
