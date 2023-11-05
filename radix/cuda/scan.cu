// ADAPTED FROM: https://github.com/jackfly/radix-sort-cuda/blob/master/cuda_implementation/scan.cu


#include "scan.h"

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
// according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf


__global__
void gpu_sum_scan_blelloch(unsigned int* const d_out,
    const unsigned int* const d_in,
    unsigned int* const d_block_sums,
    const size_t numElems)
{
    extern __shared__ unsigned int s_out[];

    unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

    s_out[threadIdx.x] = 0;
    s_out[threadIdx.x + blockDim.x] = 0;

    __syncthreads();


    unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (cpy_idx < numElems)
    {
        s_out[threadIdx.x] = d_in[cpy_idx];
        if (cpy_idx + blockDim.x < numElems)
            s_out[threadIdx.x + blockDim.x] = d_in[cpy_idx + blockDim.x];
    }

    __syncthreads();

    // Upsweep step

    unsigned int max_steps = 11; 

    unsigned int r_idx = 0;
    unsigned int l_idx = 0;
    unsigned int sum = 0; // global sum can be passed to host if needed
    unsigned int t_active = 0;
    for (int s = 0; s < max_steps; ++s)
    {
        t_active = 0;

        // calculate necessary indexes
        // right index must be (t+1) * 2^(s+1)) - 1
        r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
        if (r_idx >= 0 && r_idx < 2048)
            t_active = 1;

        if (t_active)
        {
            // left index must be r_idx - 2^s
            l_idx = r_idx - (1 << s);

            // do the actual add operation
            sum = s_out[l_idx] + s_out[r_idx];
        }
        __syncthreads();

        if (t_active)
            s_out[r_idx] = sum;
        __syncthreads();
    }

    // Copy last element (total sum of block) to block sums array
    if (threadIdx.x == 0)
    {
        d_block_sums[blockIdx.x] = s_out[r_idx];
        s_out[r_idx] = 0;
    }

    __syncthreads();

    // Downsweep step

    for (int s = max_steps - 1; s >= 0; --s)
    {
        // calculate necessary indexes
        // right index must be (t+1) * 2^(s+1)) - 1
        r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
        if (r_idx >= 0 && r_idx < 2048)
        {
            t_active = 1;
        }

        unsigned int r_cpy = 0;
        unsigned int lr_sum = 0;
        if (t_active)
        {
            // left index must be r_idx - 2^s
            l_idx = r_idx - (1 << s);

            // do the downsweep operation
            r_cpy = s_out[r_idx];
            lr_sum = s_out[l_idx] + s_out[r_idx];
        }
        __syncthreads();

        if (t_active)
        {
            s_out[l_idx] = r_cpy;
            s_out[r_idx] = lr_sum;
        }
        __syncthreads();
    }

    if (cpy_idx < numElems)
    {
        d_out[cpy_idx] = s_out[threadIdx.x];
        if (cpy_idx + blockDim.x < numElems)
            d_out[cpy_idx + blockDim.x] = s_out[threadIdx.x + blockDim.x];
    }
}

__global__
void gpu_add_block_sums(unsigned int* const d_out,
    const unsigned int* const d_in,
    unsigned int* const d_block_sums,
    const size_t numElems)
{
    unsigned int d_block_sum_val = d_block_sums[blockIdx.x];

    // Simple implementation's performance is not significantly (if at all)
    //  better than previous verbose implementation
    unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (cpy_idx < numElems)
    {
        d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
        if (cpy_idx + blockDim.x < numElems)
            d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
    }

}


__global__
void gpu_prescan(unsigned int* const d_out,
    const unsigned int* const d_in,
    unsigned int* const d_block_sums,
    const unsigned int len,
    const unsigned int shmem_sz,
    const unsigned int max_elems_per_block)
{
    // Allocated on invocation
    extern __shared__ unsigned int s_out[];

    int thid = threadIdx.x;
    int ai = thid;
    int bi = thid + blockDim.x;

    // Zero out the shared memory
    // Helpful especially when input size is not power of two
    s_out[thid] = 0;
    s_out[thid + blockDim.x] = 0;
    // If CONFLICT_FREE_OFFSET is used, shared memory
    //  must be a few more than 2 * blockDim.x
    if (thid + max_elems_per_block < shmem_sz)
        s_out[thid + max_elems_per_block] = 0;

    __syncthreads();
    
    // Copy d_in to shared memory
    // Note that d_in's elements are scattered into shared memory
    //  in light of avoiding bank conflicts
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
    if (cpy_idx < len)
    {
        s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
        if (cpy_idx + blockDim.x < len)
            s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
    }

    // Upsweep/Reduce step
    int offset = 1;
    for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_out[bi] += s_out[ai];
        }
        offset <<= 1;
    }

    // Save the total sum on the global block sums array
    // Then clear the last element on the shared memory
    if (thid == 0) 
    { 
        d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 
            + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
        s_out[max_elems_per_block - 1 
            + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
    }

    // Downsweep step
    for (int d = 1; d < max_elems_per_block; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            unsigned int temp = s_out[ai];
            s_out[ai] = s_out[bi];
            s_out[bi] += temp;
        }
    }
    __syncthreads();

    // Copy contents of shared memory to global memory
    if (cpy_idx < len)
    {
        d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
        if (cpy_idx + blockDim.x < len)
            d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
    }
}
 
void sum_scan_blelloch(unsigned int* const d_out,
    const unsigned int* const d_in,
    const size_t numElems)
{
    // Zero out d_out
    cudaMemset(d_out, 0, numElems * sizeof(unsigned int));

    // Set up number of threads and blocks
    
    unsigned int block_sz = MAX_BLOCK_SZ / 2;
    unsigned int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

    unsigned int grid_sz = numElems / max_elems_per_block;
    if (numElems % max_elems_per_block != 0) 
        grid_sz += 1;

    // Conflict free padding requires that shared memory be more than 2 * block_sz
    unsigned int shmem_sz = max_elems_per_block + ((max_elems_per_block) >> LOG_NUM_BANKS);

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    unsigned int* d_block_sums;
    cudaMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz);
    cudaMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz);

    // Sum scan data allocated to each block
    //gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
    gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_out, 
                                                                    d_in, 
                                                                    d_block_sums, 
                                                                    numElems, 
                                                                    shmem_sz,
                                                                    max_elems_per_block);


    if (grid_sz <= max_elems_per_block)
    {
        unsigned int* d_dummy_blocks_sums;
        cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int));
        cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int));
        //gpu_sum_scan_blelloch<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
        gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_block_sums, 
                                                                    d_block_sums, 
                                                                    d_dummy_blocks_sums, 
                                                                    grid_sz, 
                                                                    shmem_sz,
                                                                    max_elems_per_block);
        cudaFree(d_dummy_blocks_sums);
    }

    else
    {
        unsigned int* d_in_block_sums;
        cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * grid_sz);
        cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice);
        sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
        cudaFree(d_in_block_sums);
    }
    

    gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

    cudaFree(d_block_sums);
}