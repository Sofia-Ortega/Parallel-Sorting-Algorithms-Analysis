/**
 * -------------------- SOURCE -----------------------------------
 * Code: https://github.com/saigowri/CUDA/blob/master/quicksort.cu
 * Author: Sai Gowri
 * Date: July 15, 2016
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <chrono>

int *r_values;
int *d_values;

float dataInitTime;
float correctnessTime;

// Kernel function
__global__ static void quicksort(int *values, int N)
{
#define MAX_LEVELS 300

    int pivot, L, R;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int start[MAX_LEVELS], end[MAX_LEVELS], i = 0;

    start[idx] = idx;
    end[idx] = N - 1;

    while (idx >= 0)
    {
        L = start[idx];
        R = end[idx];
        if (L < R)
        {
            pivot = values[L];
            while (L < R)
            {
                while (values[R] >= pivot && L < R)
                {
                    R--;
                }
                if (L < R)
                {
                    values[L++] = values[R];
                }
                while (values[L] <= pivot && L < R)
                {
                    L++;
                }
                if (L < R)
                {
                    values[R--] = values[L];
                }
            }
            values[L] = pivot;
            start[idx + 1] = L + 1;
            end[idx + 1] = end[idx];
            end[idx++] = L;

            if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1])
            {
                int temp = start[idx];
                start[idx] = start[idx - 1];
                start[idx - 1] = temp;

                temp = end[idx];
                end[idx] = end[idx - 1];
                end[idx - 1] = temp;
            }
        }
        else
        {
            idx--;
        }
    }
}

int main(int argc, char **argv)
{
    CALI_CXX_MARK_FUNCTION;
    CALI_MARK_BEGIN("main");

    cudaEvent_t dataInitStart, dataInitStop, correctnessStart, correctnessStop;
    size_t size = atoi(argv[1]); // CHANGE TO CLI ARG
    printf("./quicksort starting with %d numbers...\n", size * sizeof(int));
    const int MAX_THREADS = atoi(argv[2]); // CHANGE TO CLI ARG


    std::cout << "MAX_THREADS: " << MAX_THREADS << std::endl;


    // allocate host memory
    r_values = (int *)malloc(size * sizeof(int));

    // allocate device memory
    cudaMalloc((void **)&d_values, size * sizeof(int));

    // allocate threads per block
    const unsigned int cThreadsPerBlock = 128; // CHANGE TO CLI ARG

    // Get dataset from command line
    // Generate random numbers
    cudaEventCreate(&dataInitStart);
    cudaEventCreate(&dataInitStop);
    cudaEventCreate(&correctnessStart);
    cudaEventCreate(&correctnessStop);
    
    CALI_MARK_BEGIN("dataInitTime");
    cudaEventRecord(dataInitStart, 0);
    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        r_values[i] = rand() % 100;
    }
    cudaEventRecord(dataInitStop, 0);
    cudaEventSynchronize(dataInitStop);
    CALI_MARK_END("dataInitTime");

    cudaEventElapsedTime(&dataInitTime, dataInitStart, dataInitStop);

    // Copy data from host to device
    CALI_MARK_BEGIN("comm");
    cudaMemcpy(d_values, r_values, size * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END("comm");

    // Start timer
    printf("Beginning kernel execution...\n");
    cudaThreadSynchronize();

    // Execute kernel
    auto start = std::chrono::steady_clock::now();
    CALI_MARK_BEGIN("comp");
    quicksort<<<MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock>>>(d_values, size);
    // cutilCheckMsg("Kernel execution failed...");
    CALI_MARK_END("comp");
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;


    cudaThreadSynchronize();

    printf("\nKernel execution completed in %f ms\n", diff);

    // copy data back to host
    CALI_MARK_BEGIN("comm");
    cudaMemcpy(r_values, d_values, size * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("correctness");
    cudaEventRecord(correctnessStart, 0);
    bool isSorted = true;
    for (int i = 0; i < size - 1; i++)
    {
        if (r_values[i] > r_values[i + 1])
        {
            isSorted = false;
            break;
        }

        // if (isSorted)
        // {
        //     printf("Array is sorted (LESSGO)\n");
        // }
        // else
        // {
        //     printf("Array is not sorted (womp womp)\n");
        // }
    }
    cudaEventRecord(correctnessStop, 0);
    CALI_MARK_END("correctness");

    cudaEventElapsedTime(&correctnessTime, correctnessStart, correctnessStop);
    // free memory
    cudaEventDestroy(dataInitStart);
    cudaEventDestroy(dataInitStop);
    cudaEventDestroy(correctnessStart);
    cudaEventDestroy(correctnessStop);
    free(r_values);
    cudaFree(d_values);

    // exit
    cudaThreadExit();
    cudaDeviceReset();

    CALI_MARK_END("main");
}