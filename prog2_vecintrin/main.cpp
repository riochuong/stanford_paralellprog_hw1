#include <stdio.h>
#include <algorithm>
#include <getopt.h>
#include <math.h>
#include "CS149intrin.h"
#include "logger.h"
using namespace std;

#define EXP_MAX 10

Logger CS149Logger;

void usage(const char* progname);
void initValue(float* values, int* exponents, float* output, float* gold, unsigned int N);
void absSerial(float* values, float* output, int N);
void absVector(float* values, float* output, int N);
void clampedExpSerial(float* values, int* exponents, float* output, int N);
void clampedExpVector(float* values, int* exponents, float* output, int N);
float arraySumSerial(float* values, int N);
float arraySumVector(float* values, int N);
bool verifyResult(float* values, int* exponents, float* output, float* gold, int N);

int main(int argc, char * argv[]) {
  int N = 16;
  bool printLog = false;

  // parse commandline options ////////////////////////////////////////////
  int opt;
  static struct option long_options[] = {
    {"size", 1, 0, 's'},
    {"log", 0, 0, 'l'},
    {"help", 0, 0, '?'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "s:l?", long_options, NULL)) != EOF) {

    switch (opt) {
      case 's':
        N = atoi(optarg);
        if (N <= 0) {
          printf("Error: Workload size is set to %d (<0).\n", N);
          return -1;
        }
        break;
      case 'l':
        printLog = true;
        break;
      case '?':
      default:
        usage(argv[0]);
        return 1;
    }
  }


  float* values = new float[N+VECTOR_WIDTH];
  int* exponents = new int[N+VECTOR_WIDTH];
  float* output = new float[N+VECTOR_WIDTH];
  float* gold = new float[N+VECTOR_WIDTH];
  initValue(values, exponents, output, gold, N);

  clampedExpSerial(values, exponents, gold, N);
  clampedExpVector(values, exponents, output, N);

  //absSerial(values, gold, N);
  //absVector(values, output, N);

  printf("\e[1;31mCLAMPED EXPONENT\e[0m (required) \n");
  bool clampedCorrect = verifyResult(values, exponents, output, gold, N);
  if (printLog) CS149Logger.printLog();
  CS149Logger.printStats();

  printf("************************ Result Verification *************************\n");
  if (!clampedCorrect) {
    printf("@@@ Failed!!!\n");
  } else {
    printf("Passed!!!\n");
  }

  printf("\n\e[1;31mARRAY SUM\e[0m (bonus) \n");
  if (N % VECTOR_WIDTH == 0) {
    float sumGold = arraySumSerial(values, N);
    float sumOutput = arraySumVector(values, N);
    float epsilon = 0.1;
    bool sumCorrect = abs(sumGold - sumOutput) < epsilon * 2;
    if (!sumCorrect) {
      printf("Expected %f, got %f\n.", sumGold, sumOutput);
      printf("@@@ Failed!!!\n");
    } else {
      printf("Passed!!!\n");
    }
  } else {
    printf("Must have N %% VECTOR_WIDTH == 0 for this problem (VECTOR_WIDTH is %d)\n", VECTOR_WIDTH);
  }

  delete [] values;
  delete [] exponents;
  delete [] output;
  delete [] gold;

  return 0;
}

void usage(const char* progname) {
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -s  --size <N>     Use workload size N (Default = 16)\n");
  printf("  -l  --log          Print vector unit execution log\n");
  printf("  -?  --help         This message\n");
}

void initValue(float* values, int* exponents, float* output, float* gold, unsigned int N) {

  for (unsigned int i=0; i<N+VECTOR_WIDTH; i++)
  {
    // random input values
    values[i] = -1.f + 4.f * static_cast<float>(rand()) / RAND_MAX;
    exponents[i] = rand() % EXP_MAX;
    output[i] = 0.f;
    gold[i] = 0.f;
  }

}

bool verifyResult(float* values, int* exponents, float* output, float* gold, int N) {
  int incorrect = -1;
  float epsilon = 0.00001;
  for (int i=0; i<N+VECTOR_WIDTH; i++) {
    if ( abs(output[i] - gold[i]) > epsilon ) {
      incorrect = i;
      break;
    }
  }

  if (incorrect != -1) {
    if (incorrect >= N)
      printf("You have written to out of bound value!\n");
    printf("Wrong calculation at value[%d]!\n", incorrect);
    printf("value  = ");
    for (int i=0; i<N; i++) {
      printf("% f ", values[i]);
    } printf("\n");

    printf("exp    = ");
    for (int i=0; i<N; i++) {
      printf("% 9d ", exponents[i]);
    } printf("\n");

    printf("output = ");
    for (int i=0; i<N; i++) {
      printf("% f ", output[i]);
    } printf("\n");

    printf("gold   = ");
    for (int i=0; i<N; i++) {
      printf("% f ", gold[i]);
    } printf("\n");
    return false;
  }
  printf("Results matched with answer!\n");
  return true;
}

// computes the absolute value of all elements in the input array
// values, stores result in output
void absSerial(float* values, float* output, int N) {
  for (int i=0; i<N; i++) {
    float x = values[i];
    if (x < 0) {
      output[i] = -x;
    } else {
      output[i] = x;
    }
  }
}


// implementation of absSerial() above, but it is vectorized using CS149 intrinsics
void absVector(float* values, float* output, int N) {
  __cs149_vec_float x;
  __cs149_vec_float result;
  __cs149_vec_float zero = _cs149_vset_float(0.f);
  __cs149_mask maskAll, maskIsNegative, maskIsNotNegative;

//  Note: Take a careful look at this loop indexing.  This example
//  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
//  Why is that the case?
  for (int i=0; i<N; i+=VECTOR_WIDTH) {

    // All ones
    maskAll = _cs149_init_ones();

    // All zeros
    maskIsNegative = _cs149_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _cs149_vload_float(x, values+i, maskAll);               // x = values[i];

    // Set mask according to predicate
    _cs149_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _cs149_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _cs149_mask_not(maskIsNegative);     // } else {

    // Execute instruction ("else" clause)
    _cs149_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _cs149_vstore_float(output+i, result, maskAll);
  }
}


// accepts an array of values and an array of exponents
//
// For each element, compute values[i]^exponents[i] and clamp value to
// 9.999.  Store result in output.
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
  for (int i=0; i<N; i++) {
    float x = values[i];
    int y = exponents[i];
    if (y == 0) {
      output[i] = 1.f;
    } else {
      float result = x;
      int count = y - 1;
      while (count > 0) {
        result *= x;
        count--;
      }
      if (result > 9.999999f) {
        result = 9.999999f;
      }
      output[i] = result;
    }
  }
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {

  //
  // CS149 STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  int num_iterations = N / VECTOR_WIDTH;
  // use this for vector that is not completely fit 
  int remainings = N - (VECTOR_WIDTH * num_iterations);
  if (remainings) num_iterations += 1;
  __cs149_vec_int vect_zeros = _cs149_vset_int(0);
  __cs149_vec_int vect_ones = _cs149_vset_int(1);
  __cs149_vec_float vect_max = _cs149_vset_float(9.999999f);
  __cs149_mask copy_mask = _cs149_init_ones(); 
  int custom_mask [VECTOR_WIDTH] = {0};
  for (int i = 0; i < num_iterations; i++){
      int start_idx = i * VECTOR_WIDTH;
      __cs149_mask vect_masks = _cs149_init_ones();
      // fix the mask for overflow value that not fitting
      if (i == (num_iterations - 1) && remainings){
          for (int i = 0; i < remainings; i++) {custom_mask[i] = 1;}
          __cs149_vec_int tmp_vect;
          _cs149_vload_int(tmp_vect, custom_mask, vect_masks);
          // update vect masks 
          _cs149_veq_int(vect_masks, tmp_vect, vect_ones, vect_masks);
          copy_mask = _cs149_mask_and(copy_mask, vect_masks);
      }
      __cs149_mask reverse_mask;
      __cs149_vec_float vect_vals;
      __cs149_vec_int vect_exp = _cs149_vset_int(0);
      __cs149_vec_float vect_results = _cs149_vset_float(1.0); 
      __cs149_mask vect_mask_max_check;
      _cs149_vload_float(vect_vals, &values[start_idx], vect_masks);
      _cs149_vload_int(vect_exp, &exponents[start_idx], vect_masks);
      // update initial vect masks with anything already zeros
      _cs149_vgt_int(vect_masks, vect_exp, vect_zeros, vect_masks);

      while(_cs149_cntbits(vect_masks)){
          _cs149_vmult_float(vect_results, vect_vals, vect_results, vect_masks);
          _cs149_vsub_int(vect_exp, vect_exp, vect_ones, vect_masks);
          // update vect masks per clamp
          // mask check if values greater than max 9.999999f
          _cs149_vgt_float(vect_mask_max_check, vect_results, vect_max, vect_masks);
          vect_mask_max_check = _cs149_mask_and(vect_mask_max_check, vect_masks);
         // update vect masks per exponet
          // if (i == num_iterations - 1){
          //   printf("vectmasks %d %f %d %f\n", 
          //     vect_mask_max_check.value[0], 
          //     vect_results.value[0], 
          //     vect_masks.value[0], 
          //     vect_max.value[0]);
          // } 
          // set max value and these will be frozen on next iteration
          _cs149_vset_float(vect_results, 9.999999f, vect_mask_max_check);
          
          // if (i == num_iterations - 1){
          //   printf("vectmasks after %d %f %d %f\n", 
          //     vect_mask_max_check.value[0], 
          //     vect_results.value[0], 
          //     vect_masks.value[0], 
          //     vect_max.value[0]);
          // } 
          
          _cs149_vgt_int(vect_masks, vect_exp, vect_zeros, vect_masks);
          // smaller than 9.999f && exponent greater than 0
          reverse_mask = _cs149_mask_not(vect_mask_max_check);
          vect_masks = _cs149_mask_and(vect_masks, reverse_mask);
      }
      // copy result back to output
      _cs149_vstore_float(&output[start_idx], vect_results, copy_mask);
  }
  
}

// returns the sum of all elements in values
float arraySumSerial(float* values, int N) {
  float sum = 0;
  for (int i=0; i<N; i++) {
    sum += values[i];
  }

  return sum;
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {
  
  //
  // CS149 STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  
  for (int i=0; i<N; i+=VECTOR_WIDTH) {

  }

  return 0.0;
}

