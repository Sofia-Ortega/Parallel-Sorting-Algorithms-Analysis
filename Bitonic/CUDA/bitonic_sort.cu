/*
 * ADAPTED FROM LAB 3 OF CSCE 435
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../../InputGeneration/inputgen.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* main_time = "main";
const char* data_init = "data_init";
const char* comm_small = "comm_small_1";
const char* comm_small_2 = "comm_small_2";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correct_check = "correct_check";
const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

// Store results in these variables.
float effective_bandwidth_gb_s;
float bitonic_sort_step_time;
float cudaMemcpy_host_to_device_time;
float cudaMemcpy_device_to_host_time;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  // CALI_MARK_BEGIN(comp_small);

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }

  // CALI_MARK_END(comp_small);
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  CALI_MARK_BEGIN(main);
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);

  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  
  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  CALI_MARK_BEGIN(comm_small_1);

  cudaEventRecord(start1, 0);

  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  
  cudaEventRecord(stop1, 0);
  cudaEventSynchronize(stop1);

  CALI_MARK_END(comm_small_1);
  CALI_MARK_END(cudaMemcpy_host_to_device);

  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, start1, stop1);

  printf("CUDA Host to Device Mem Copy Time: %.4f ms\n", cudaMemcpy_host_to_device_time);
  
  cudaEventDestroy(start1);
  cudaEventDestroy(stop1);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  int kernel_count = 0;

  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);

  CALI_MARK_BEGIN(bitonic_sort_step_region);

  cudaEventRecord(start2, 0);

  CALI_MARK_BEGIN(comp_large);

  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      kernel_count++;
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }

  CALI_MARK_END(comp_large);

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);

  cudaDeviceSynchronize();
  
  CALI_MARK_END(bitonic_sort_step_region);
  
  cudaEventElapsedTime(&bitonic_sort_step_time, start2, stop2);
  
  printf("CUDA Bitonic Sort Time: %.4f ms\n", bitonic_sort_step_time);
  
  cudaEventDestroy(start2);
  cudaEventDestroy(stop2);

  cudaEvent_t start3, stop3;
  cudaEventCreate(&start3);
  cudaEventCreate(&stop3);

  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  CALI_MARK_BEGIN(comm_small_2);

  cudaEventRecord(start3, 0);

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop3, 0);
  cudaEventSynchronize(stop3);

  CALI_MARK_END(comm_small_2);
  CALI_MARK_END(cudaMemcpy_device_to_host);

  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, start3, stop3);
  
  printf("CUDA Device to Host Mem Copy Time: %.4f ms\n", cudaMemcpy_device_to_host_time);
  
  // Calculate effective bandwidth
  float hold = (size * 6 * kernel_count) / bitonic_sort_step_time;
  effective_bandwidth_gb_s = hold / 1e6;

  printf("Effective Bandwidth: %.4f GBps\n", effective_bandwidth_gb_s);
  
  cudaEventDestroy(start3);
  cudaEventDestroy(stop3);

  cudaFree(dev_values);
  CALI_MARK_END(main_time);
}

int main(int argc, char *argv[])
{
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  int option = atoi(argv[3]);

  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;

  CALI_MARK_BEGIN(data_init);

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  // array_fill(values, NUM_VALS);
  if (option == 0) {
    // random
    array_fill(values, NUM_VALS);
  } 
  else if (option == 1) {
    // sorted
    for (int i = 0; i < NUM_VALS; i++) {
      values[i] = i;
    }
  }
  else if (option == 2) {
    // reverse sorted
    for (int i = 0; i < NUM_VALS; i++) {
      values[i] = NUM_VALS - i;
    }
  }
  else if (option == 3) {
    // perturb 1& of values
    for (int i = 0; i < NUM_VALS; i++) {
      values[i] = i;
    }

    int num_perturb = NUM_VALS / 100;
    for (int i = 0; i < num_perturb; i++) {
      int index = rand() % NUM_VALS;
      values[index] = rand() % NUM_VALS;
    }
  }

  CALI_MARK_END(data_init);

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();

  CALI_MARK_BEGIN(correct_check);

  for (int i = 0; i < NUM_VALS - 1; i++) {
    if (values[i] > values[i + 1]) {
      printf("ERROR: Array not sorted correctly!\n");
      break;
    }
  }

  CALI_MARK_END(correct_check);
  
  print_elapsed(start, stop);


  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("num_vals", NUM_VALS);
  adiak::value("program_name", "cuda_bitonic_sort");
  adiak::value("datatype_size", sizeof(float));
  adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
  adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
  adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
  adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}