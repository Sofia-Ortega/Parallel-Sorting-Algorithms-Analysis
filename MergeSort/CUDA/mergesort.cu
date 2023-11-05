/*
 * merge_kernel.cu
 * 
 *  MergeSort Implementation run in parallel threads
 *   on the GPU through the nVidia CUDA Framework
 * 
 * Jim Kukunas and James Devine
 *
 */
#include <stdio.h>
#include <stdlib.h>
//#include <cutil_inline.h>
//#include "merge_kernel.cu"
 
#ifndef _MERGE_KERNEL_CU_
#define _MERGE_KERNEL_CU_

//#define NUM    512

__device__ inline
  void Merge(int* values, int* results, int l, int r, int u)
{
  int i,j,k;
  i=l; j=r; k=l;
  while (i<r && j<u) { 
    if (values[i]<=values[j]) {results[k]=values[i]; i++;} 
    else {results[k]=values[j]; j++;}
    k++;
  }
  
  while (i<r) { 
    results[k]=values[i]; i++; k++;
  }
  
  while (j<u) { 
    results[k]=values[j]; j++; k++;
  }
  for (k=l; k<u; k++) { 
    values[k]=results[k]; 
  }
}

__global__ static void MergeSort(int * values, int* results, int nums)
{
    extern __shared__ int shared[];

    const unsigned int tid = threadIdx.x;
    int k,u,i;
 
    // Copy input to shared mem.
    shared[tid] = values[tid];
    
    __syncthreads();
    
    k = 1;
    while(k < nums)
    {
        i = 0;
        while(i+k <= nums)
        {
            u = i+k*2;
            if(u > nums)
            {
                u = nums+1;
            }
            Merge(shared, results, i, i+k, u);
            i = i+k*2;
        }
        k = k*2;
        __syncthreads();
    }
    
    values[tid] = shared[tid];
}

int main(int argc, char** argv)
{
	int nums = atoi(argv[2]);

    //int values[NUM];
	int *values = new int[nums];

    /* initialize a random dataset */
	srand(10);
    for(int i = 0; i < nums; i++)
    {
        values[i] = rand();
    }

    int *dvalues, *results;
	
	cudaMalloc((void**)&dvalues, sizeof(int) * nums);
    cudaMemcpy(dvalues, values, sizeof(int) * nums, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&results, sizeof(int) * nums);
    cudaMemcpy(results,values , sizeof(int)* nums, cudaMemcpyHostToDevice);

    MergeSort<<<1, nums, sizeof(int) * nums*2>>>(dvalues, results, nums);

	/*
	for(int i=0; i<NUM; ++i) printf("%d, ", values[i]);
	printf("\n");
	*/

	cudaFree(dvalues);
    cudaMemcpy(values, results, sizeof(int)*nums, cudaMemcpyDeviceToHost);
    cudaFree(results);

	/*
	for(int i=0; i<NUM; ++i) printf("%d, ", values[i]);
	printf("\n");
	*/

    bool passed = true;
    for(int i = 1; i < nums; i++)
    {
        if (values[i-1] > values[i])
        {
            passed = false;
        }
    }
    printf( "Test %s\n", passed ? "PASSED" : "FAILED");
	delete[] values;
    cudaThreadExit();
}

#endif 
