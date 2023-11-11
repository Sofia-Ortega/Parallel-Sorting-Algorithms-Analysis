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
float commSmallTime;
float commLargeTime;
float compSmallTime;
float compLargeTime;

const char *mainRegion = "main";
const char *parallel = "parallel";
const char *sequential = "sequential";
const char *genValuesTime = "data_init";
const char *barrier = "barrier";
const char *correctness = "correctness_check";
const char *comp = "comp";
const char *compSmall = "comp_small";
const char *compLarge = "comp_large";
const char *commRegion = "comm";
const char *commSmall = "comm_small";
const char *commLarge = "comm_large";

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
    CALI_MARK_BEGIN(mainRegion);

    cudaEvent_t dataInitStart, dataInitStop, correctnessStart, correctnessStop;
    cudaEvent_t compSmallStart, compSmallStop, commSmallStart, commSmallStop;
    cudaEvent_t compLargeStart, compLargeStop, commLargeStart, commLargeStop;
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
    cudaEventCreate(&commSmallStart);
    cudaEventCreate(&commSmallStop);
    cudaEventCreate(&commLargeStart);
    cudaEventCreate(&commLargeStop);
    cudaEventCreate(&compSmallStart);
    cudaEventCreate(&compSmallStop);
    cudaEventCreate(&compLargeStart);
    cudaEventCreate(&compLargeStop);
    cudaEventCreate(&correctnessStart);
    cudaEventCreate(&correctnessStop);

    CALI_MARK_BEGIN(genValuesTime);
    cudaEventRecord(dataInitStart, 0);
    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        r_values[i] = rand() % 100;
    }
    cudaEventRecord(dataInitStop, 0);
    cudaEventSynchronize(dataInitStop);
    CALI_MARK_END(genValuesTime);

    cudaEventElapsedTime(&dataInitTime, dataInitStart, dataInitStop);

    // Copy data from host to device
    CALI_MARK_BEGIN(commRegion);
    CALI_MARK_BEGIN(commLarge);
    cudaEventRecord(commLargeStart, 0);
    cudaMemcpy(d_values, r_values, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(commLargeStop, 0);
    cudaEventSynchronize(commLargeStop);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(commRegion);

    cudaEventElapsedTime(&commLargeTime, commLargeStart, commLargeStop);

    // Start timer
    printf("Beginning kernel execution...\n");
    cudaThreadSynchronize();

    // Execute kernel
    CALI_MARK_BEGIN(comp);
    cudaEventRecord(compLargeStart, 0);
    quicksort<<<MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock>>>(d_values, size);
    cudaEventRecord(compLargeStop, 0);
    cudaEventSynchronize(compLargeStop);
    CALI_MARK_END(comp);

    cudaEventElapsedTime(&compLargeTime, compLargeStart, compLargeStop);

    cudaThreadSynchronize();

    printf("\nKernel execution completed in %f ms\n", compLargeTime);

    // copy data back to host
    CALI_MARK_BEGIN(commRegion);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(r_values, d_values, size * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(commRegion);

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
    }
    cudaEventRecord(correctnessStop, 0);
    CALI_MARK_END("correctness");

    cudaEventElapsedTime(&correctnessTime, correctnessStart, correctnessStop);

    if (isSorted)
    {
        printf("Array is sorted (LESSGO)\n");
    }
    else
    {
        printf("Array is not sorted (womp womp)\n");
    }

    // Print out all times
    printf("Data init time: %f ms\n", dataInitTime);
    printf("Comm large time: %f ms\n", commLargeTime);
    printf("Comp large time: %f ms\n", compLargeTime);
    printf("Correctness check time: %f ms\n", correctnessTime);
    // free memory
    cudaEventDestroy(dataInitStart);
    cudaEventDestroy(dataInitStop);
    cudaEventDestroy(commSmallStart);
    cudaEventDestroy(commSmallStop);
    cudaEventDestroy(commLargeStart);
    cudaEventDestroy(commLargeStop);
    cudaEventDestroy(compSmallStart);
    cudaEventDestroy(compSmallStop);
    cudaEventDestroy(compLargeStart);
    cudaEventDestroy(compLargeStop);
    cudaEventDestroy(correctnessStart);
    cudaEventDestroy(correctnessStop);
    free(r_values);
    cudaFree(d_values);

    // exit
    cudaThreadExit();
    cudaDeviceReset();

    CALI_MARK_END(mainRegion);
}