
// ADAPTED FROM: https://github.com/jackfly/radix-sort-cuda/blob/master/cuda_implementation/main.cu

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

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

using namespace std;

const char* main = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

const char* radix_sort_region = "radix_sort";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";


void radixsort_gpu(unsigned int* h_in, unsigned int num, unsigned int num_threads, bool printArray = false)
{
    unsigned int* out_gpu = new unsigned int[num];
    
    unsigned int* d_in;
    unsigned int* d_out;
    cudaMalloc(&d_in, sizeof(unsigned int) * num);
    cudaMalloc(&d_out, sizeof(unsigned int) * num);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    radix_sort(d_out, d_in, num, num_threads);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(out_gpu, d_out, sizeof(unsigned int) * num, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    if(printArray) {
      printf("------------- Sorted: ----------------\n");
      for(int i = 0; i < num; i++) {
        printf("%i\n", out_gpu[i]);
      }
      printf("--------------\n");
    }

    // check if sorted
    CALI_MARK_BEGIN(correctness_check);
    bool isSorted = true;
    for(int i = 1; i < num; i++) {
      if (out_gpu[i - 1] > out_gpu[i]) {
        isSorted = false;
        break;
      }
    }
    CALI_MARK_END(correctness_check);

    if(isSorted) {
      printf("[SUCCESS] Output Sorted\n");
    } else {
      printf("[FAILED] Output NOT Sorted\n");
    }

    cudaFree(d_out);
    cudaFree(d_in);

    delete[] out_gpu;
}

int main(int argc, char** argv)
{

    // argv:
    // 0          1            2                 3
    // radix_cuda num_threasds num_vals_to_sort  [optional: printArray]
    struct timespec start, stop;

    CALI_CXX_MARK_FUNCTION;


    // get user input
    if(argc != 3 && argc != 4) {
      printf("Incorrect argument usage\n");
      printf("radix_cuda num_threads num_vals_to_sort [optional: print_array]\n");
      return -1;
    }
    
    int num_threads = atoi(argv[1]);
    int n_values = atoi(argv[2]);
    bool printArray = false;

    if(argc == 4) {
      printArray = atoi(argv[3]);
    }

    printf("Sorting %i values with %i threads\n", n_values, num_threads);

    // create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(main);

    // initialize local array
    unsigned int* numbers = new unsigned int[n_values];
    for(int i = 0; i < n_values; i++) {
      numbers[i] = (rand() % 10000) + 1;
    }


    // print array
    if(printArray) {
      for(int i = 0; i < n_values; i++) {
        printf("%i\n", numbers[i]);
      }
    }

    // sorting
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    radixsort_gpu(numbers, n_values, num_threads, printArray);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    double dt = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;    // in microseconds
    printf("@time of CUDA run:\t\t\t[%.3f] microseconds\n", dt);

    delete[] numbers;
    CALI_MARK_END(main);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "radix"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n_values); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_threads); // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_threads / n_values ); // The number of CUDA blocks 
    adiak::value("group_num", 23); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "online") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten"). 
    
    // Flush Caliper output
    mgr.stop();
    mgr.flush();


}