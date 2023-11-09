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
#include <cutil_inline.h>

int *r_values;
int *d_values;

// Kernel function
__global__ static void quicksort(int *values)
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



    printf("./quicksort starting with %d numbers...\n", N);
    unsigned int hTimer;
    size_t size = atoi(argv[1]); // CHANGE TO CLI ARG
    const int MAX_THREADS = atoi(argv[2]); // CHANGE TO CLI ARG

    // allocate host memory
    r_values = (int *)malloc(size);

    // allocate device memory
    cutilSafeCall(cudaMalloc((void **)&d_values, size));

    // allocate threads per block
    const unsigned int cThreadsPerBlock = 128; // CHANGE TO CLI ARG

    // Get dataset from command line
    // Generate random numbers
    
    CALI_MARK_BEGIN("dataInitTime");
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        r_values[i] = rand() % 100;
    }
    CALI_MARK_END("dataInitTime");

    // Copy data from host to device
    CALI_MARK_BEGIN("comm");
    cutilSafeCall(cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice));
    CALI_MARK_END("comm");

    // Start timer
    printf("Beginning kernel execution...\n");
    cutilCheckError(cutCreateTimer(&hTimer));
    cutilCheckError(cudaThreadSynchronize());
    cutilCheckError(cutResetTimer(hTimer));
    cutilCheckError(cutStartTimer(hTimer));

    // Execute kernel
    auto start = std::chrono::steady_clock::now();
    CALI_MARK_BEGIN("comp");
    quicksort<<<MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock>>>(d_values);
    cutilCheckMsg("Kernel execution failed...");
    CALI_MARK_END("comp");


    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError(cutStopTimer(hTimer));
    double gpuTime = cutGetTimerValue(hTimer);

    printf("\nKernel execution completed in %f ms\n", gpuTime);

    // copy data back to host
    CALI_MARK_BEGIN("comm");
    cutilSafeCall(cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost));
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("correctness");
    bool isSorted = true;
    for (int i = 0; i < size - 1; i++)
    {
        if (r_values[i] > r_values[i + 1])
        {
            isSorted = false;
            break;
        }
    }
    CALI_MARK_END("correctness");
    // free memory
    free(r_values);
    cutilSafeCall(cudaFree(d_values));

    // exit
    cudaThreadExit();
    cutilExit(argc, argv);

    CALI_MARK_END("main");
}