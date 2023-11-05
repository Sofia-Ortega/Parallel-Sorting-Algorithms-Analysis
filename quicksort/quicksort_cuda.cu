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

#define MAX_THREADS 128
#define N 512

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
    printf("./quicksort starting with %d numbers...\n", N);
    unsigned int hTimer;
    size_t size = N * sizeof(int);

    // allocate host memory
    r_values = (int *)malloc(size);

    // allocate device memory
    cutilSafeCall(cudaMalloc((void **)&d_values, size));

    // allocate threads per block
    const unsigned int cThreadsPerBlock = 128;

    // Get dataset from command line
    if (argc > 1)
    {
        for (int i = 0; i < N; i++)
        {
            r_values[i] = atoi(argv[i + 1]);
        }
    }
    else
    {
        // Generate random numbers
        srand(time(NULL));
        for (int i = 0; i < N; i++)
        {
            r_values[i] = rand() % 100;
        }
    }

    // Copy data from host to device
    cutilSafeCall(cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice));

    // Start timer
    printf("Beginning kernel execution...\n");
    cutilCheckError(cutCreateTimer(&hTimer));
    cutilCheckError(cudaThreadSynchronize());
    cutilCheckError(cutResetTimer(hTimer));
    cutilCheckError(cutStartTimer(hTimer));

    // Execute kernel
    quicksort<<<MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock>>>(d_values);
    cutilCheckMsg("Kernel execution failed...");

    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError(cutStopTimer(hTimer));
    double gpuTime = cutGetTimerValue(hTimer);

    printf("\nKernel execution completed in %f ms\n", gpuTime);

    // copy data back to host
    cutilSafeCall(cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost));

    // test print
    for (int i = 0; i < N; i++) {
        printf("%d ", r_values[i]);
    }
    printf("\n");

    // free memory
    free(r_values);
    cutilSafeCall(cudaFree(d_values));

    // exit
    cudaThreadExit();
    cutilExit(argc, argv);
}