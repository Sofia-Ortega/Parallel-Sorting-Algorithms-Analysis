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

    cudaEvent_t dataInitStart, dataInitStop, correctnessStart, correctnessStop;
    cudaEvent_t compSmallStart, compSmallStop, commSmallStart, commSmallStop;
    cudaEvent_t compLargeStart, compLargeStop, commLargeStart, commLargeStop;
    size_t size = atoi(argv[1]); // CHANGE TO CLI ARG
    printf("./quicksort starting with %d numbers...\n", size * sizeof(int));
    const int MAX_THREADS = atoi(argv[2]); // CHANGE TO CLI ARG
    const int option = atoi(argv[3]);      // CHANGE TO CLI ARG

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
    if (option == 0) // random
    {
        for (int i = 0; i < size; i++)
        {
            r_values[i] = rand() % 100;
        }
    } else if (option == 1) { //sorted
        for (int i = 0; i < size; i++) {
            r_values[i] = i;
        }
    } else if (option == 2) { // reverse
        for (int i = 0; i < size; i++) {
            r_values[i] = size - i;
        }
    } else if (option == 3) { // 1% perturbed
        for (int i = 0; i < size; i++) {
            r_values[i] = i;
        }
        for (int i = 0; i < size / 100; i++) {
            int index = rand() % size;
            r_values[index] = size - index;
        }
    }
    cudaEventRecord(dataInitStop, 0);
    cudaEventSynchronize(dataInitStop);
    CALI_MARK_END(genValuesTime);

    cudaEventElapsedTime(&dataInitTime, dataInitStart, dataInitStop);

    // Copy data from host to device
    cudaEventRecord(commLargeStart, 0);

    CALI_MARK_BEGIN(commRegion);
    CALI_MARK_BEGIN(commLarge);
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(d_values, r_values, size * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END(commLarge);
    CALI_MARK_END(commRegion);

    cudaEventRecord(commLargeStop, 0);
    cudaEventSynchronize(commLargeStop);

    cudaEventElapsedTime(&commLargeTime, commLargeStart, commLargeStop);

    // Start timer
    printf("Beginning kernel execution...\n");
    cudaThreadSynchronize();

    // Execute kernel
    cudaEventRecord(compLargeStart, 0);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(compLarge);
    quicksort<<<MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock>>>(d_values, size);
    CALI_MARK_END(compLarge);
    CALI_MARK_END(comp);

    cudaEventRecord(compLargeStop, 0);
    cudaEventSynchronize(compLargeStop);

    cudaEventElapsedTime(&compLargeTime, compLargeStart, compLargeStop);

    cudaThreadSynchronize();

    printf("\nKernel execution completed in %f ms\n", compLargeTime);

    // copy data back to host
    CALI_MARK_BEGIN(commRegion);
    CALI_MARK_BEGIN(commLarge);
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(r_values, d_values, size * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END(commLarge);
    CALI_MARK_END(commRegion);

    cudaEventRecord(correctnessStart, 0);

    CALI_MARK_BEGIN(correctness);
    bool isSorted = true;
    for (int i = 0; i < size - 1; i++)
    {
        if (r_values[i] > r_values[i + 1])
        {
            isSorted = false;
            break;
        }
    }
    CALI_MARK_END(correctness);
    cudaEventRecord(correctnessStop, 0);

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

    cali::ConfigManager mgr;
    mgr.start();

    adiak::init(NULL);
    adiak::launchdate();                                        // launch date of the job
    adiak::libraries();                                         // Libraries used
    adiak::cmdline();                                           // Command line used to launch the job
    adiak::clustername();                                       // Name of the cluster
    adiak::value("Algorithm", "quicksort");                     // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");                   // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                            // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));                // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size);                            // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", cThreadsPerBlock);              // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", MAX_THREADS / cThreadsPerBlock); // The number of CUDA blocks
    adiak::value("group_num", 23);                              // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "online");            // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output
    mgr.stop();
    mgr.flush();

    // exit
    cudaThreadExit();
    cudaDeviceReset();
}