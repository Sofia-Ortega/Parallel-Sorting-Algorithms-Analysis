#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <chrono>

/**
 * mergesort.cu
 * a one-file c++ / cuda program for performing mergesort on the GPU
 * While the program execution is fairly slow, most of its runnning time
 *  is spent allocating memory on the GPU.
 * For a more complex program that performs many calculations,
 *  running on the GPU may provide a significant boost in performance
 * 
 * This code was written by Kevin Albert and was obtained from his GitHub
 * at https://github.com/54kevinalbert/gpu-mergesort. I (Will Thompson)
 * modified the code for the assignment.
 */

// data[], size, threads, blocks, 
void mergesort(double*, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(double*, double*, long, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(double*, double*, long, long, long);

#define min(a, b) (a < b ? a : b)

int main(int argc, char** argv) {

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    //
    // Read numbers from stdin
    //
	int size = atoi(argv[2]);;
    double* data = new double[size];
	srand(10);
	for(int i=0; i<size; ++i) data[i] = (double)(rand() % 100);

    // merge-sort the data
	auto start = std::chrono::steady_clock::now();
    mergesort(data, size, threadsPerBlock, blocksPerGrid);
	auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	std::cout << "CUDA Mergesort Time: " << end.count() << " ms" << std::endl;

	bool isSorted = true;
	for (int i=0; i<size-1; ++i)
	{
		if(data[i] > data[i+1])
		{
			isSorted = false;
			break;
		}
	}
	std::cout << "The list is sorted: " << isSorted << std::endl;

}

void mergesort(double* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    double* D_data;
    double* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
	cudaMalloc((void**) &D_data, size * sizeof(double));
    cudaMalloc((void**) &D_swp, size * sizeof(double));

    // Copy from our input list into the first array
    cudaMemcpy(D_data, data, size * sizeof(double), cudaMemcpyHostToDevice);
 
    //
    // Copy the thread / block info to the GPU as well
    //
	cudaMalloc((void**) &D_threads, sizeof(dim3));
    cudaMalloc((void**) &D_blocks, sizeof(dim3));

	cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    double* A = D_data;
    double* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    //
    // Get the list back from the GPU
    //
    cudaMemcpy(data, A, size * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free the GPU memory
	cudaFree(A);
    cudaFree(B);
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(double* source, double* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(double* source, double* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}
