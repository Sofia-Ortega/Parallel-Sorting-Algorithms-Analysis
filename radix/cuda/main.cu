
// ADAPTED FROM: https://github.com/jackfly/radix-sort-cuda/blob/master/cuda_implementation/main.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string> 
#include <sstream>
#include <math.h>
#include <time.h>

#include "radix_sort.h"
#include "inputgen.h"

using namespace std;

void radixsort_gpu(unsigned int* h_in, unsigned int num)
{
    unsigned int* out_gpu = new unsigned int[num];
    
    unsigned int* d_in;
    unsigned int* d_out;
    cudaMalloc(&d_in, sizeof(unsigned int) * num);
    cudaMalloc(&d_out, sizeof(unsigned int) * num);
    cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num, cudaMemcpyHostToDevice);

    radix_sort(d_out, d_in, num);

    cudaMemcpy(out_gpu, d_out, sizeof(unsigned int) * num, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);

    delete[] out_gpu;
}

int main(int argc, char** argv)
{
    // argv:

    // 0          1                 2
    // radix_cuda num_vals_to_sort  [optional: printArray]
    struct timespec start, stop;

    if(argc != 2 || argc != 3) {
      printf("Incorrect argument usage\n");
      printf("radix_cuda num_vals_to_sort [optional: print_array]\n");
    }
    

    // initialize local array
    int n_values = atoi(argv[1]);
    bool printArray = false;

    if(argc == 3) {
      printArray = atoi(argv[2]);
    }

    unsigned int* numbers = new unsigned int[n_values];
    fillValsRandParallel(numbers, n_values, 10);

    printf("Sorting %i values", n_values);

    // print array
    if(printArray) {
      for(int i = 0; i < n_values; i++) {
        printf("%i\n", numbers[i]);
      }
    }



    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    radixsort_gpu(numbers, n_values);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    double dt = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;    // in microseconds
    printf("@time of CUDA run:\t\t\t[%.3f] microseconds\n", dt);

    // print array
    if(printArray) {
      for(int i = 0; i < n_values; i++) {
        printf("%i\n", numbers[i]);
      }
    }

    delete[] numbers;

}