
// ADAPTED FROM: https://github.com/jackfly/radix-sort-cuda/blob/master/cuda_implementation/main.cu

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

#include "sort.h"
using namespace std;

void radixsort_gpu(unsigned int* h_in, unsigned int num)
{
    unsigned int* out_gpu = new unsigned int[num];
    
    unsigned int* d_in;
    unsigned int* d_out;
    cudaMalloc(&d_in, sizeof(unsigned int) * num);
    cudaMalloc(&d_out, sizeof(unsigned int) * num);
    cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num, cudaMemcpyHostToDevice);

    radix_sort(d_out, d_in, num);

    cudaMemcpy(out_gpu, d_out, sizeof(unsigned int) * num, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);

    delete[] out_gpu;
}

int main()
{
    struct timespec start, stop;
    for (int i = 0; i <= 18;)
    {
        i = i + 2;
        int num = pow(2,i);
        int linecount = 0;
        string filename = "../RandomNumbers/" + std::to_string(i) + ".txt";
        unsigned int* numbers = new unsigned int[num];
        //int numbers[num];
        //Create an input file stream
        fstream file(filename);
        for(int m=0; m<num; m++)
        {
            file >> numbers[linecount];
            linecount++;
        }
        //Close the file stream
        file.close();

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        radixsort_gpu(numbers, num);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        double dt = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;    // in microseconds
        printf("@time of CUDA run:\t\t\t[%.3f] microseconds\n", dt);

        delete[] numbers;

    }
}