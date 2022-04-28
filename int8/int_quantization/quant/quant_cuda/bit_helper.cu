#define FLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float*>(x))

__device__ __forceinline__ unsigned int round_log_quantization(unsigned int target) {
    if (target == 0) {
      return target;
    }
    unsigned int rand_prob = 1 << 22;
    unsigned int add_r = target + rand_prob; //nearest round
    //unsigned int add_r = target; //no nearest round
    //unsigned int sign = add_r >> 31;
    //unsigned int exp = (add_r >> 23) & 0x000000ff;
    //unsigned int quantized = (sign << 31) | (exp << 23);
    
    //return quantized;
    return add_r & 0xff800000; //this should be faster
}