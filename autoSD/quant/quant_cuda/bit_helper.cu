#define FLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float*>(x))

//floatsd8
__device__ __forceinline__ unsigned int round_floatsd8_dynamic_332(unsigned int target, int bias) { //3+3+2
    if (target == 0) {
      return target;
    }
    unsigned int rand_prob = 1 << 18; 
    unsigned int add_r = target+rand_prob; //nearest round
    //unsigned int add_r = target; //no nearest round
    unsigned int sign = add_r >> 31;
    unsigned int mant = (add_r >> 19) & 0x0000000f; //only need first four bit of mantissa
    unsigned int exp = add_r << 1 >> 1 >> 23; // bias -10 / -10 + 127 = 117
    unsigned int quantized = 0;
    unsigned int max_exp = (127-bias+11);
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){
      quantized = (sign << 31) | (max_exp << 23) | (2 << 19);
    }
    else if (exp < min_exp){
      quantized = (sign << 31) | (min_exp << 23);
    }
    else if (mant == 0){ //1
      quantized = (sign << 31) | (exp << 23);
    }
    else if (mant == 1){ //1.0625    
      if (exp < min_exp+4){ // 2^0~3
        quantized = (sign << 31) | (exp << 23); //1.0625 -> 1
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (mant << 19);
      }
    }
    else if (mant == 2){ //1.125
      if (exp < min_exp+3){ // 2^0~2
        quantized = (sign << 31) | (exp << 23); //1.125 -> 1
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (mant << 19);
      }
    }
    else if ((mant == 3) || (mant == 4) || (mant == 5)){ //1.25
      if (exp < min_exp+2){ // 2^0~1
        quantized = (sign << 31) | (exp << 23); //1.25 -> 1
      }
      else if (exp == max_exp){ // 2^11
        quantized = (sign << 31) | (exp << 23) | (2 << 19); //1.25 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (4 << 19);
      }
    }
    else if ((mant == 6) || (mant == 7) || (mant == 8) || (mant == 9)){ //1.5
      if (exp == min_exp){ // 2^0
        quantized = (sign << 31) | (exp << 23); //1.5 -> 1
      }
      else if (exp == max_exp-1){ // 2^10
        quantized = (sign << 31) | (exp << 23) | (12 << 19); //1.25 -> 1.75
      }
      else if (exp == max_exp){ // 2^11
        quantized = (sign << 31) | (exp << 23) | (2 << 19); //1.25 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (8 << 19);
      }
    }
    else if ((mant == 10) || (mant == 11) || (mant == 12)){ //1.75
      if (exp < min_exp+2){ // 2^0~1
        quantized = (sign << 31) | ((exp+1) << 23); //1.75 -> 1 / exp ++
      }
      else if (exp == max_exp){ // 2^11
        quantized = (sign << 31) | (exp << 23) | (2 << 19); //1.75 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (12 << 19);
      }
    }
    else { // 13/14/15 => 1.875
      if (exp < min_exp+3){ // 2^0~2
        quantized = (sign << 31) | ((exp+1) << 23); //1.875 -> 1 / exp ++
      }
      else if (exp == max_exp){ // 2^11
        quantized = (sign << 31) | (exp << 23) | (2 << 19); //1.875 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (14 << 19);
      }
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_floatsd8_dynamic_233(unsigned int target, int bias) { //2+3+3
    if (target == 0) {
      return target;
    }
    unsigned int rand_prob = 1 << 17; 
    unsigned int add_r = target+rand_prob; //nearest round
    //unsigned int add_r = target; //no nearest round
    unsigned int sign = add_r >> 31;
    unsigned int mant = (add_r >> 18) & 0x0000001f; //only need first five bits of mantissa
    unsigned int exp = add_r << 1 >> 24; // bias -10 / -10 + 127 = 117
    unsigned int quantized = 0;
    unsigned int max_exp = (127-bias+8);
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){
      quantized = (sign << 31) | (max_exp << 23) | (4 << 18);
    }
    else if (exp < min_exp){
      quantized = (sign << 31) | (min_exp << 23);
    }
    else if (mant == 0){ //1
      quantized = (sign << 31) | (exp << 23);
    }
    else if (mant == 1){ //1.03125  
      if (exp < min_exp+5){ // 2^0~4
        quantized = (sign << 31) | (exp << 23); //1.03125 -> 1
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (mant << 18);
      }
    }
    else if (mant == 2){ //1.0625
      if (exp < min_exp+4){ // 2^0~3
        quantized = (sign << 31) | (exp << 23); //1.0625 -> 1
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (mant << 18);
      }
    }
    else if ((mant == 3) || (mant == 4) || (mant == 5)){ //1.125
      if (exp < min_exp+3){ // 2^0~2
        quantized = (sign << 31) | (exp << 23); //1.125 -> 1
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (4 << 18);
      }
    }
    else if ((mant > 5) && (mant < 12)){ //1.25
      if (exp < min_exp+3){ // 2^0~2
        quantized = (sign << 31) | (exp << 23); //1.25 -> 1
      }
      else if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.25 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (8 << 18);
      }
    }
    else if ((mant > 11) && (mant < 20)){ //1.5
      if (exp < min_exp+2){ // 2^0~1
        quantized = (sign << 31) | (exp << 23); //1.5 -> 1
      }
      else if (exp == max_exp-1){ // 2^7
        quantized = (sign << 31) | (exp << 23) | (8 << 18); //1.5 -> 1.25
      }
      else if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.5 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (16 << 18);
      }
    }
    else if ((mant > 19) && (mant < 26)){ //1.75
      if (exp < min_exp+2){ // 2^0~1
        quantized = (sign << 31) | ((exp+1) << 23); //1.75 -> 2
      }
      else if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.75 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (24 << 18);
      }
    }
    else if ((mant > 25) && (mant < 29)){ //1.875
      if (exp < min_exp+3){ // 2^0~2
        quantized = (sign << 31) | ((exp+1) << 23); //1.875 -> 2
      }
      else if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.875 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (28 << 18);
      }
    }
    else if ((mant == 29) || (mant == 30)){ //1.9375
      if (exp < min_exp+4){ // 2^0~3
        quantized = (sign << 31) | ((exp+1) << 23); //1.9375 -> 2
      }
      else if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.9375 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (30 << 18);
      }
    }
    else { //31 => 2
      if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.9375 -> 1.125
      }
      else {
        quantized = (sign << 31) | ((exp+1) << 23);
      }
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_floatsd8_dynamic_323(unsigned int target, int bias) { //3+2+3
    if (target == 0) {
      return target;
    }
    unsigned int add_r = target; //no nearest round
    unsigned int sign = add_r >> 31;
    unsigned int mant = (add_r >> 19) & 0x0000000f; //only need first five bits of mantissa
    unsigned int exp = add_r << 1 >> 24; // bias -10 / -10 + 127 = 117
    unsigned int quantized = 0;
    unsigned int max_exp = 127;
    unsigned int min_exp = 116;

    if (exp > max_exp){
      quantized = (sign << 31) | (max_exp << 23) | (4 << 18);
    }
    else if (exp < min_exp){
      quantized = 0;
    }
    else if (mant == 0){ //1
      quantized = (sign << 31) | (exp << 23);
    }
    else if (mant == 1){ //1.0625   
      if (exp < 120){ // 2^0~4
        quantized = (sign << 31) | (exp << 23); //1.03125 -> 1
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (mant << 18);
      }
    }
    else if (mant == 2){ //1.125
      if (exp < 119){ // 2^0~3
        quantized = (sign << 31) | (exp << 23); //1.0625 -> 1
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (mant << 18);
      }
    }
    else if ((mant == 3) || (mant == 4) || (mant == 5)){ //1.25
      if (exp < 119){ // 2^0~2
        quantized = (sign << 31) | (exp << 23); //1.125 -> 1
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (4 << 18);
      }
    }
    else if ((mant > 5) && (mant < 10)){ //1.5
      if (exp < 118){ // 2^0~2
        quantized = (sign << 31) | (exp << 23); //1.25 -> 1
      }
      else if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.25 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (8 << 18);
      }
    }
    else if ((mant > 9) && (mant < 13)){ //1.75
      if (exp < 118){ // 2^0~1
        quantized = (sign << 31) | ((exp+1) << 23); //1.5 -> 1
      }
      else if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.5 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (12 << 18);
      }
    }
    else if ((mant == 13) || (mant == 14)){ //1.875
      if (exp < 119){ // 2^0~1
        quantized = (sign << 31) | ((exp+1) << 23); //1.75 -> 2
      }
      else if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.75 -> 1.125
      }
      else {
        quantized = (sign << 31) | (exp << 23) | (14 << 18);
      }
    }
    else { //31 => 2
      if (exp == max_exp){ // 2^8
        quantized = (sign << 31) | (exp << 23) | (4 << 18); //1.9375 -> 1.125
      }
      else {
        quantized = (sign << 31) | ((exp+1) << 23);
      }
    }
    return quantized;
}

//fp8
__device__ __forceinline__ unsigned int round_fp8_152_float_nearest_dynamic(unsigned int target, int bias) { //dynamic offset
    if (target == 0) {
      return target;
    }
    unsigned int mask = (1 << 21) - 1;
    unsigned int rand_prob = 1 << 20;
    unsigned int add_r = (target+rand_prob) & ~mask;
    unsigned int sign = add_r >> 31 << 31;
    unsigned int mant = add_r << 9 >> 9;
    unsigned int exp = add_r << 1 >> 24;
    unsigned int quantized = 0;
    unsigned int max_exp = (158-bias); //bias=24. max=134/min=103
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){
      //quantized = sign | (max_exp << 23) | 3 << 21;
      quantized = sign | (255 << 23);
    }
    else if (exp < min_exp-1){
      quantized = 0; //flush to zero
    }
    else if (exp < min_exp){ // == min_exp-1
      quantized = sign | (min_exp << 23) | 1 << 21; //minimum nonzero value
    }
    else if (exp == min_exp){ // == min_exp
      if (mant == 0){ //allocate to zero
        quantized = sign | (min_exp << 23) | 1 << 21;
      }
      else{
        quantized = sign | (min_exp << 23) | mant;
      }
    }
    else {
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_fp8_152_clip_float_nearest_dynamic(unsigned int target, int bias) { //dynamic offset
    if (target == 0) {
      return target;
    }
    unsigned int mask = (1 << 21) - 1;
    unsigned int rand_prob = 1 << 20;
    //unsigned int rand_prob = 0;
    unsigned int add_r = (target+rand_prob) & ~mask;
    unsigned int sign = add_r >> 31 << 31;
    unsigned int mant = add_r << 9 >> 9;
    unsigned int exp = add_r << 1 >> 24;
    unsigned int quantized = 0;
    unsigned int max_exp = (158-bias); //bias=24. max=134/min=103
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){
      quantized = sign | (max_exp << 23) | 3 << 21;
      //quantized = sign | (255 << 23);
    }
    else if (exp < min_exp-1){
      quantized = 0; //flush to zero
    }
    else if (exp < min_exp){ // == min_exp-1
      quantized = sign | (min_exp << 23) | 1 << 21; //minimum nonzero value
    }
    else if (exp == min_exp){ // == min_exp
      if (mant == 0){ //allocate to zero
        quantized = sign | (min_exp << 23) | 1 << 21;
      }
      else{
        quantized = sign | (min_exp << 23) | mant;
      }
    }
    else {
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_fp8_143_float_nearest_dynamic(unsigned int target, int bias) { //dynamic offset
    if (target == 0) {
      return target;
    }
    unsigned int mask = (1 << 20) - 1;
    unsigned int rand_prob = 1 << 19;
    unsigned int add_r = (target+rand_prob) & ~mask;
    unsigned int sign = add_r >> 31 << 31;
    unsigned int mant = add_r << 9 >> 9;
    unsigned int exp = add_r << 1 >> 24;
    unsigned int quantized = 0;
    unsigned int max_exp = (142-bias); //bias=24. max=134/min=103
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){ // max value = 1.875 * 2^max_exp
      quantized = sign | (255 << 23);
    }
    else if (exp < min_exp-1){ //min value = 1 * 2^min_exp
      quantized = 0;
    }
    else if (exp < min_exp){ // == min_exp-1
      quantized = sign | (min_exp << 23) | 1 << 20; //minimum nonzero value
    }
    else if (exp == min_exp){ // == min_exp
      if (mant == 0){ //allocate to zero
        quantized = sign | (min_exp << 23) | 1 << 20;
      }
      else{
        quantized = sign | (min_exp << 23) | mant;
      }
    }
    else {
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_fp8_143_clip_float_nearest_dynamic(unsigned int target, int bias) { //dynamic offset
    if (target == 0) {
      return target;
    }
    unsigned int mask = (1 << 20) - 1;
    unsigned int rand_prob = 1 << 19;
    unsigned int add_r = (target+rand_prob) & ~mask;
    //unsigned int sign = add_r >> 31 << 31;
    unsigned int sign = add_r & 0x80000000;
    //unsigned int mant = add_r << 9 >> 9;
    unsigned int mant = add_r & 0x007FFFFF;
    //unsigned int exp = add_r << 1 >> 24;
    unsigned int exp = (add_r & 0x7FFFFFFF) >> 23;
    unsigned int quantized;
    unsigned int max_exp = (142-bias); //bias=24. max=134/min=103
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){ // max value = 1.875 * 2^max_exp
      quantized = sign | (max_exp << 23) | 7 << 20;
    }
    else if (exp < min_exp-1){ //min value = 1 * 2^min_exp
      quantized = 0;
    }
    else if (exp < min_exp){ // == min_exp-1
      quantized = sign | (min_exp << 23) | 1 << 20; //minimum nonzero value
    }
    else if (exp == min_exp){ // == min_exp
      if (mant == 0){ //allocate to zero
        quantized = sign | (min_exp << 23) | 1 << 20;
      }
      else{
        quantized = sign | (min_exp << 23) | mant;
      }
    }
    else { //max_exp ~ min_exp+1 => 2*15*8
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_fp8_134_float_nearest_dynamic(unsigned int target, int bias) { //dynamic offset
    if (target == 0) {
      return target;
    }
    unsigned int mask = (1 << 19) - 1;
    unsigned int rand_prob = 1 << 18;
    unsigned int add_r = (target+rand_prob) & ~mask;
    unsigned int sign = add_r >> 31 << 31;
    unsigned int mant = add_r << 9 >> 9;
    unsigned int exp = add_r << 1 >> 24;
    unsigned int quantized = 0;
    unsigned int max_exp = (134-bias); //bias=24. max=134/min=103
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){ // max value = 1.9375 * 2^max_exp
      quantized = sign | (255 << 23);
    }
    else if (exp < min_exp-1){ //min value = 1 * 2^min_exp
      quantized = 0;
    }
    else if (exp < min_exp){ // == min_exp-1
      quantized = sign | (min_exp << 23) | 1 << 19; //minimum nonzero value
    }
    else if (exp == min_exp){ // == min_exp
      if (mant == 0){ //allocate to zero
        quantized = sign | (min_exp << 23) | 1 << 19;
      }
      else{
        quantized = sign | (min_exp << 23) | mant;
      }
    }
    else {
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_fp8_134_clip_float_nearest_dynamic(unsigned int target, int bias) { //dynamic offset
    if (target == 0) {
      return target;
    }
    unsigned int mask = (1 << 19) - 1;
    unsigned int rand_prob = 1 << 18;
    unsigned int add_r = (target+rand_prob) & ~mask;
    unsigned int sign = add_r >> 31 << 31;
    unsigned int mant = add_r << 9 >> 9;
    unsigned int exp = add_r << 1 >> 24;
    unsigned int quantized = 0;
    unsigned int max_exp = (134-bias); //bias=24. max=134/min=103
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){ // max value = 1.9375 * 2^max_exp
      quantized = sign | (max_exp << 23) | 15 << 19;
    }
    else if (exp < min_exp-1){ //min value = 1 * 2^min_exp
      quantized = 0;
    }
    else if (exp < min_exp){ // == min_exp-1
      quantized = sign | (min_exp << 23) | 1 << 19; //minimum nonzero value
    }
    else if (exp == min_exp){ // == min_exp
      if (mant == 0){ //allocate to zero
        quantized = sign | (min_exp << 23) | 1 << 19;
      }
      else{
        quantized = sign | (min_exp << 23) | mant;
      }
    }
    else {
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}

//floatsd4
__device__ __forceinline__ unsigned int round_floatsd4_float_dynamic(unsigned int target, int bias) {
    if (target == 0) {
      return target;
    }
    //unsigned int rand_prob = 1 << 22; //1000000000 
    //unsigned int add_r = target + rand_prob; //nearest round
    unsigned int add_r = target; //no nearest round
    unsigned int sign = add_r >> 31;
    unsigned int exp = (add_r >> 23) & 0x00ff; // bias -10 / -10 + 15 = 5
    unsigned int quantized = 0;
    unsigned int max_exp = (127-bias+5);
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){
      quantized = (sign << 31) | (max_exp << 23);
    }
    else if (exp < min_exp){
      quantized = 0;
    }
    else {
      quantized = (sign << 31) | (exp << 23);
    }
    
    return quantized;
}

__device__ __forceinline__ unsigned int round_floatsd4_ex_float_dynamic(unsigned int target, int bias) {
    if (target == 0) {
      return target;
    }
    unsigned int rand_prob = 1 << 22;
    unsigned int add_r = target + rand_prob; //nearest round
    //uinsigned int add_r = target; //no nearest round
    unsigned int sign = add_r >> 31;
    unsigned int exp = (add_r >> 23) & 0x000000ff; // bias -10 / -10 + 15 = 5
    unsigned int quantized = 0;
    unsigned int max_exp = (127-bias+6);
    unsigned int min_exp = (127-bias);

    if (exp > max_exp){
      quantized = (sign << 31) | (max_exp << 23);
    }
    else if (exp < min_exp){
      quantized = 0;
    }
    else {
      quantized = (sign << 31) | (exp << 23);
    }
    
    return quantized;
}

//fp
__device__ __forceinline__ unsigned int round_fp_float_nearest_dynamic(unsigned int target, int bias,
                                                                       int exp_bit, int mantissa_bit) { //dynamic offset
    if (target == 0) {
      return target;
    }
    int shift = 23 - mantissa_bit;
    unsigned int mask = (1 << shift) - 1;
    unsigned int rand_prob = 1 << (shift - 1);
    unsigned int add_r = (target+rand_prob) & ~mask;
    unsigned int sign = add_r >> 31 << 31;
    unsigned int mant = add_r << 9 >> 9;
    unsigned int exp = add_r << 1 >> 24;
    unsigned int quantized = 0;
    unsigned int min_exp = (127-bias);
    unsigned int max_exp = 0;

    if (exp_bit == 0){
      max_exp = min_exp;
    }
    else if (mantissa_bit > 0) {
      max_exp = min_exp + (2 << (exp_bit-1)) - 1; //FP8_134 => max = min + 2 << 2 - 1 = min + 7 
    }
    else {
      max_exp = min_exp + (2 << (exp_bit-1)) - 2; //FloatSD4_ex => max = min + 2 << 2 - 2 = min + 6
    }
    /* for debug
    printf("max exp:%d\n", max_exp);
    printf("exp_bit:%d\n", exp_bit);
    printf("mantissa_bit:%d\n", mantissa_bit);
    printf("%d\n", 2 << (exp_bit-1));*/

    if (exp > max_exp){ // max value = infinite
      quantized = sign | (255 << 23);
    }
    else if (exp < min_exp-1){ //flush to zero
      quantized = 0;
    }
    else if (exp == min_exp-1){ //saturate to minimum nonzero value
      if (mantissa_bit > 0) {
        quantized = sign | (min_exp << 23) | 1 << shift; //minimum nonzero value
      }
      else {
        quantized = sign | (min_exp << 23);
      }
    }
    else if (exp == min_exp){ // == min_exp
      if (mantissa_bit > 0) {
        if (mant == 0){ //allocate to zero => saturate to minimum nonzero value
          quantized = sign | (min_exp << 23) | 1 << shift; //minimum nonzero value
        }
        else{
          quantized = sign | (min_exp << 23) | mant;
        }
      }
      else {
        quantized = sign | (min_exp << 23);
      }
    }
    else {
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_fp_clip_float_nearest_dynamic(unsigned int target, int bias,
                                                                            int exp_bit, int mantissa_bit) { //dynamic offset
    if (target == 0) {
      return target;
    }
    int shift = 23 - mantissa_bit;
    unsigned int mask = (1 << shift) - 1;
    unsigned int rand_prob = 1 << (shift - 1);
    unsigned int add_r = (target+rand_prob) & ~mask;
    unsigned int sign = add_r >> 31 << 31;
    unsigned int mant = add_r << 9 >> 9;
    unsigned int exp = add_r << 1 >> 24;
    unsigned int quantized = 0;
    unsigned int min_exp = (127-bias);
    unsigned int max_exp = 0;

    if (exp_bit == 0){
      max_exp = min_exp;
    }
    else if (mantissa_bit > 0) {
      max_exp = min_exp + (2 << (exp_bit-1)) - 1; //FP8_134 => max = min + 2 << 2 - 1 = min + 7 
    }
    else {
      max_exp = min_exp + (2 << (exp_bit-1)) - 2; //FloatSD4_ex => max = min + 2 << 2 - 2 = min + 6
    }
    /* for debug
    printf("max exp:%d\n", max_exp);
    printf("exp_bit:%d\n", exp_bit);
    printf("mantissa_bit:%d\n", mantissa_bit);
    printf("%d\n", 2 << (exp_bit-1));*/

    if (exp > max_exp){ // max value = max mantissa * 2^max_exp
      if (mantissa_bit > 0) {
        quantized = sign | (max_exp << 23) | (0x007fffff  & ~mask);
      }
      else {
        quantized = sign | (max_exp << 23);
      }
    }
    else if (exp < min_exp-1){ //flush to zero
      quantized = 0;
    }
    else if (exp == min_exp-1){ //saturate to minimum nonzero value
      if (mantissa_bit > 0) {
        quantized = sign | (min_exp << 23) | 1 << shift; //minimum nonzero value
      }
      else {
        quantized = sign | (min_exp << 23);
      }
    }
    else if (exp == min_exp){ // == min_exp
      if (mantissa_bit > 0) {
        if (mant == 0){ //allocate to zero => saturate to minimum nonzero value
          quantized = sign | (min_exp << 23) | 1 << shift; //minimum nonzero value
        }
        else{
          quantized = sign | (min_exp << 23) | mant;
        }
      }
      else {
        quantized = sign | (min_exp << 23);
      }
    }
    else {
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}

__device__ __forceinline__ unsigned int round_fp_clip_float_stochastic_dynamic(unsigned int target, int bias,
                                                                               int exp_bit, int mantissa_bit,
                                                                               unsigned int rand_prob) { //dynamic offset
    if (target == 0) {
      return target;
    }
    int shift = 23 - mantissa_bit;
    unsigned int mask = (1 << shift) - 1;
    unsigned int add_r = (target+(rand_prob & mask)) & ~mask;
    unsigned int sign = add_r >> 31 << 31;
    unsigned int mant = add_r << 9 >> 9;
    unsigned int exp = add_r << 1 >> 24;
    unsigned int quantized = 0;
    unsigned int min_exp = (127-bias);
    unsigned int max_exp = 0;

    if (exp_bit == 0){
      max_exp = min_exp;
    }
    else if (mantissa_bit > 0) {
      max_exp = min_exp + (2 << (exp_bit-1)) - 1; //FP8_134 => max = min + 2 << 2 - 1 = min + 7 
    }
    else {
      max_exp = min_exp + (2 << (exp_bit-1)) - 2; //FloatSD4_ex => max = min + 2 << 2 - 2 = min + 6
    }

    if (exp > max_exp){ // max value = max mantissa * 2^max_exp
      if (mantissa_bit > 0) {
        quantized = sign | (max_exp << 23) | (0x007fffff  & ~mask);
      }
      else {
        quantized = sign | (max_exp << 23);
      }
    }
    else if (exp < min_exp-1){ //flush to zero
      quantized = 0;
    }
    else if (exp == min_exp-1){ //saturate to minimum nonzero value
      if (mantissa_bit > 0) {
        quantized = sign | (min_exp << 23) | 1 << shift; //minimum nonzero value
      }
      else {
        quantized = sign | (min_exp << 23);
      }
    }
    else if (exp == min_exp){ // == min_exp
      if (mantissa_bit > 0) {
        if (mant == 0){ //allocate to zero => saturate to minimum nonzero value
          quantized = sign | (min_exp << 23) | 1 << shift; //minimum nonzero value
        }
        else{
          quantized = sign | (min_exp << 23) | mant;
        }
      }
      else {
        quantized = sign | (min_exp << 23);
      }
    }
    else {
      quantized = sign | (exp << 23) | mant;
    }
    return quantized;
}