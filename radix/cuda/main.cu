
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

#include "scan.h"

using namespace std;

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



#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5


#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

__global__ void gpu_radix_sort_local(unsigned int* d_out_sorted,
    unsigned int* d_prefix_sums,
    unsigned int* d_block_sums,
    unsigned int input_shift_width,
    unsigned int* d_in,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{

    extern __shared__ unsigned int shmem[];
    unsigned int* s_data = shmem;
    // s_mask_out[] will be scanned in place
    unsigned int s_mask_out_len = max_elems_per_block + (max_elems_per_block >> LOG_NUM_BANKS);
    unsigned int* s_mask_out = &s_data[max_elems_per_block];
    unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];

    unsigned int thid = threadIdx.x;

    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
    if (cpy_idx < d_in_len)
        s_data[thid] = d_in[cpy_idx];
    else
        s_data[thid] = 0;

    __syncthreads();

    unsigned int t_data = s_data[thid];
    unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;

    for (unsigned int i = 0; i < 4; ++i)
    {
        // Zero out s_mask_out
        s_mask_out[thid] = 0;
        
        if (thid + max_elems_per_block < s_mask_out_len)
            s_mask_out[thid + max_elems_per_block] = 0;
        __syncthreads();

        // build bit mask output
        bool val_equals_i = false;
        if (cpy_idx < d_in_len)
        {
            val_equals_i = t_2bit_extract == i;
            s_mask_out[thid + CONFLICT_FREE_OFFSET(thid)] = val_equals_i;
        }
        __syncthreads();

        // scan bit mask output
        // Upsweep/Reduce step
        bool t_active = thid < (blockDim.x / 2);
        int offset = 1;
        for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
        {
            __syncthreads();

            if (t_active && (thid < d))
            {
                int ai = offset * ((thid << 1) + 1) - 1;
                int bi = offset * ((thid << 1) + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                s_mask_out[bi] += s_mask_out[ai];
            }
            offset <<= 1;
        }

        // Save the total sum on the global block sums array
        // Then clear the last element on the shared memory
        if (thid == 0)
        {
            //unsigned int total_sum_idx = (unsigned int) fmin();
            unsigned int total_sum = s_mask_out[max_elems_per_block - 1
                + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
            s_mask_out[max_elems_per_block - 1
                + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
        }
        __syncthreads();

        // Downsweep step
        for (int d = 1; d < max_elems_per_block; d <<= 1)
        {
            offset >>= 1;
            __syncthreads();

            if (t_active && (thid < d))
            {
                int ai = offset * ((thid << 1) + 1) - 1;
                int bi = offset * ((thid << 1) + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                unsigned int temp = s_mask_out[ai];
                s_mask_out[ai] = s_mask_out[bi];
                s_mask_out[bi] += temp;
            }
        }
        __syncthreads();

        if (val_equals_i && (cpy_idx < d_in_len))
        {
            s_merged_scan_mask_out[thid] = s_mask_out[thid + CONFLICT_FREE_OFFSET(thid)];
        }
        __syncthreads();
    }
    
    __syncthreads();

    // Scan mask output sums
    if (thid == 0)
    {
        unsigned int run_sum = 0;
        for (unsigned int i = 0; i < 4; ++i)
        {
            s_scan_mask_out_sums[i] = run_sum;
            run_sum += s_mask_out_sums[i];
        }
    }
    __syncthreads();

    if (cpy_idx < d_in_len)
    {
        // Calculate the new indices of the input elements for sorting
        unsigned int new_pos = s_merged_scan_mask_out[thid] + s_scan_mask_out_sums[t_2bit_extract];
        //if (new_ai >= 1024)
        //    new_ai = 0;
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];
        
        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        s_data[new_pos] = t_data;
        s_merged_scan_mask_out[new_pos] = t_prefix_sum;
        
        __syncthreads();

        // copy block-wise sort results to global 
        // then copy block-wise prefix sum results to global memory
        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];
        d_out_sorted[cpy_idx] = s_data[thid];
    }
}

__global__ void gpu_glbl_shuffle(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len)
    {
        unsigned int t_data = d_in[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];
        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x]
            + t_prefix_sum;
        __syncthreads();
        d_out[data_glbl_pos] = t_data;
    }
}

void radix_sort(unsigned int* const d_out,
    unsigned int* const d_in,
    unsigned int d_in_len,
    unsigned int num_threads)
{
    unsigned int block_sz = num_threads;
    unsigned int max_elems_per_block = block_sz;
    unsigned int grid_sz = d_in_len / max_elems_per_block;
    
    if (d_in_len % max_elems_per_block != 0)
        grid_sz += 1;
    // initialize the prefix sum variable
    unsigned int* d_prefix_sums;
    unsigned int d_prefix_sums_len = d_in_len;
    cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * d_prefix_sums_len);
    cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_prefix_sums_len);

    unsigned int* d_block_sums;
    unsigned int d_block_sums_len = 4 * grid_sz; // 4-way split
    cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int* d_scan_block_sums;
    cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int s_data_len = max_elems_per_block;
    unsigned int s_mask_out_len = max_elems_per_block + (max_elems_per_block / NUM_BANKS);
    unsigned int s_merged_scan_mask_out_len = max_elems_per_block;
    unsigned int s_mask_out_sums_len = 4; // 4-way split
    unsigned int s_scan_mask_out_sums_len = 4;
    unsigned int shmem_sz = (s_data_len 
                            + s_mask_out_len
                            + s_merged_scan_mask_out_len
                            + s_mask_out_sums_len
                            + s_scan_mask_out_sums_len)
                            * sizeof(unsigned int);


    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    CALI_MARK_BEGIN("comp");
    for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 2)
    {
        CALI_MARK_BEGIN("comp_large");
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_out, 
                                                                d_prefix_sums, 
                                                                d_block_sums, 
                                                                shift_width, 
                                                                d_in, 
                                                                d_in_len, 
                                                                max_elems_per_block);

        CALI_MARK_END("comp_large");
        // scan global block sum array
        sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

        // scatter/shuffle block-wise sorted array to final positions
        CALI_MARK_BEGIN("comp_small");
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in, 
                                                    d_out, 
                                                    d_scan_block_sums, 
                                                    d_prefix_sums, 
                                                    shift_width, 
                                                    d_in_len, 
                                                    max_elems_per_block);
        CALI_MARK_END("comp_small");
    }
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    cudaMemcpy(d_out, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice);
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    cudaFree(d_scan_block_sums);
    cudaFree(d_block_sums);
    cudaFree(d_prefix_sums);
}

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
    adiak::value("implementation_source", "online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten"). 
    
    // Flush Caliper output
    mgr.stop();
    mgr.flush();


}