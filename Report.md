# Parallel Computing (CSCE 435) Group project

## Table of Contents
1. [Introduction](#introduction)
2. [Contributors](#contributors)
3. [Algorithm Description + Pseudo-code](#algorithm-description--pseudo-code)
    - [Parallel Radix Sort (MPI + CUDA)](#parallel-radix-sort-mpi--cuda)
    - [Parallel Merge Sort (MPI + CUDA)](#parallel-merge-sort-mpi--cuda)
    - [Parallel Quick Sort (MPI + CUDA)](#parallel-quick-sort-mpi--cuda)
4. [Algorithm Documentation](#algorithm-documentation)
    - [Radix Sort (Sofia)](#radix-sort)
        - [MPI](#mpi)
        - [CUDA](#cuda)
    - [Mergesort](#mergesort)
        - [MPI](#mpi-1)
        - [CUDA](#cuda-1)
    - [Bitonic Sort](#bitonic-sort)
        - [MPI](#mpi-2)
        - [CUDA](#cuda-2)
    - [Quicksort](#quicksort)
        - [MPI](#mpi-3)
        - [CUDA](#cuda-3)
5. [Performance Evaluation & Report](#performance-evaluation--report)
    - [Algorithm Comparisons](#algorithm-comparisons)
        - [Strong Scaling (Sofia)](#strong-scaling)
        - [Weak Scaling](#weak-scaling)
        - [Speedup (Sofia)](#speedup)
    - [Radix Sort Analysis and PerformanceEval/Plots (Sofia)](#radix-sort-analysis-and-performanceevalplots)
        - [MPI](#mpi-4)
        - [CUDA](#cuda-4)
    - [Mergesort Analysis and PerformanceEval/Plots](#mergesort-analysis-and-performanceevalplots)
        - [Strong Scaling](#strong-scaling-1)
            - [MPI](#mpi-5)
            - [CUDA](#cuda-5)
        - [Speedup](#strong-scaling-speedup)
            - [MPI](#mpi-6)
            - [CUDA](#cuda-6)
        - [Weak Scaling](#weak-scaling-1)
            - [MPI](#mpi-7)
            - [CUDA](#cuda-7)
    - [Bitonic Sort PerformanceEval/Plots](#bitonic-sort-performanceevalplots)
        - [Strong Scaling](#strong-scaling-2)
            - [MPI](#mpi-8)
            - [CUDA](#cuda-8)
        - [Speedup](#speedup-1)
            - [MPI](#mpi-9)
            - [CUDA](#cuda-9)
        - [Weak Scaling](#weak-scaling-2)
            - [MPI](#mpi-10)
            - [CUDA](#cuda-10)
    - [Quick Sort PerformanceEval/Plots](#quick-sort-performanceevalplots)
        - [Strong Scaling](#strong-scaling-3)
            - [MPI](#mpi-11)
            - [CUDA](#cuda-11)
        - [Speedup](#strong-scaling-speedup-1)
            - [MPI](#mpi-12)
            - [CUDA](#cuda-12)
        - [Weak Scaling](#weak-scaling-3)
            - [MPI](#mpi-13)
            - [CUDA](#cuda-13)

# Introduction

This project involves comparing the performance of various parallel sorting algorithms using different array inputs. The inputs include sorted, reversed, and randomly selected values. The algorithms are implemented using both MPI and CUDA to measure their scalability and performance. The algorithms studied are:

- Parallel Radix Sort (MPI + CUDA)
- Parallel Merge Sort (MPI + CUDA)
- Parallel Quick Sort (MPI + CUDA)
- Bitonic Sort (MPI + CUDA)

The performance evaluation includes strong scaling, weak scaling, and speedup analysis. The results are presented with detailed plots and analysis for each algorithm and input type.


## Contributors:
**Sofia Ortega**:
- Radix Sort Implementation
- Algorithm Comparisions for Strong Scaling 
- Algorithm Comparison for Speedup 

**Will Thompson**: 
- Merge Sort Implementation
- Algorithm Comparisons for Weak Scaling 

**Krithivel Ramesh**: 
- Bitonic Sort Implementation

**Sriabhinandan Venkataraman**: 
- Quick Sort Implementation


## Algorithm Description + Pseudo-code 

The sources referenced to implement the algorithms

### Parallel Radix Sort (MPI + CUDA)

  Note: Radix sort only works with integers

  Pseudo-code: [Source](https://cs.stackexchange.com/questions/6871/how-does-the-parallel-radix-sort-work)
  
```
parallel_for part in 0..K-1
  for i in indexes(part)
    bucket = compute_bucket(a[i])
    Cnt[part][bucket]++

base = 0
for bucket in 0..R-1
  for part in 0..K-1
    Cnt[part][bucket] += base
    base = Cnt[part][bucket]

parallel_for part in 0..K-1
  for i in indexes(part)
    bucket = compute_bucket(a[i])
    out[Cnt[part][bucket]++] = a[i]
```
  

### Parallel Merge Sort (MPI + CUDA)
  
  Pseudo-code: [Source](https://en.wikipedia.org/wiki/Merge_sort)

```
// Sort elements lo through hi (exclusive) of array A.
algorithm mergesort(A, lo, hi) is
if lo+1 < hi then  // Two or more elements.
mid := ⌊(lo + hi) / 2⌋
fork mergesort(A, lo, mid)
mergesort(A, mid, hi)
join
merge(A, lo, mid, hi)
```
  
### Parallel Quick Sort (MPI + CUDA)
  
  Pseudo-code: [Source](https://www3.cs.stonybrook.edu/~rezaul/Spring-2019/CSE613/CSE613-lecture-8.pdf)

```
partition(A[q : r], x):
  n = r - q + 1
  if n==1 return q

  array B[0: n-1], lessthan[0: n-1], greaterthan[0: n-1]
  parallel for i = 0 to n -1:
      B[i] = A[q + i]
      if B[i] < x then lessthan[i] = 1 else lessthan[i] = 0
      if B[i] > x then greaterthan[i] = 1 else greaterthan[i] = 0
  lessthan[0: n-1] = prefixsum(lessthan[0: n-1])
  greaterthan[0: n-1] = prefixsum(greaterthan[0: n-1])
  k = q + lessthan[n-1], A[k] = x
  parallel for i = 0 to n -1:
      if B[i] < x then A[q + lessthan[i] - 1] = B[i]
      else if B[i] > x then A[k + greaterthan[i]] = B[i]
  return k
  
quicksort(A[q : r]):
  select random element x from A[q : r]
  k = partition(A[q : r], x)
  fork quicksort(A[q : k - 1])
  quicksort(A[k + 1 : r])
  sync

where each quicksort call and each partition call are done in parallel.
```



# Algorithm Documentation
This section will review how each algorithm was implemented in the project and how to run each algorithm

## Radix Sort

### MPI

#### Summary

Implemented Radix Sort with MPI. 
We initialize the array with `InputGeneration/inputgen.cpp` to generate the input in parallel.
Each MPI process receives a sub array of the input to sort independently. 
Throughout the runtime, the MPI processes communicate the counts and elements between each other in order for each process to know where to correctly place their own elements in relation to the entire array. 


#### Running

1. Run `. build.sh`

2. Run sbatch:

```
sbatch radix.grace_job  <t> <n>
```

- `t`: number of threads 
- `n`: length of array you want to sort 

#### Adapted From 


* Source Code: https://github.com/jackfly/radix-sort-cuda/tree/master
 * Author: Jack Liu
 * Date: December 18, 2017

### CUDA

#### Summary

This MPI implementation was created with CUDA. 
In this implementation of radix, multiple passes are done on the array.
For each pass, we separate the array between blocks, which performs radix sort on its own individual subset of the array. 
`gpu_glbl_shuffle` is called to coordinate so that each element is inserted in the correct position of the overall array after all the blocks are finished

#### Running

1. Run `. build.sh`

2. Run sbatch:

```
sbatch radix.grace_job  <n> <p>
```

- `n`: how many numbers you want to sort 
- `p`: number of processes

#### Adapted From 


* Source Code: https://github.com/ym720/p_radix_sort_mpi/tree/master/p_radix_sort_mpi

 * Report: https://andreask.cs.illinois.edu/Teaching/HPCFall2012/Projects/yourii-report.pdf
 * Author: Yourii Martiak
 * University: New York University
 * Date: December 20, 2012


## Mergesort


### MPI

#### Summary

The input gets generated in parallel using the inputgeneration code. The array is scattered across the different processors and then sorted on each processor using the serial mergesort algorithm. The processors for a binary tree structure where the right child will send its data to the left child to be merged into one array. This process is repeated until the entire array is merged and sorted.


#### Running

1. Run `. build.sh`

2. Run sbatch:

```
sbatch mergesort.grace_job <t> <n> <option>
```

- `t`: number of processors
- `n`: length of array you want to sort
- `option`: 0 for random array, 1 for sorted, and 2 for reverse sorted

#### Adapted From 


* Source Code: http://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html

### CUDA

#### Summary

The CUDA implementation was much more difficult to get working. It seems mergesort is slightly easier to go about in MPI. I believe the CUDA code works by sorting lots of small arrays across many threads. It then inceases the numbers that each thread is resposible for and repeats the process with fewer threads. It keeps repeating this process until the list is sorted.

#### Running

1. Run `. build.sh`

2. Run sbatch:

```
sbatch mergesort.grace_job <n> <p> <option>
```

- `n`: how many numbers you want to sort 
- `p`: number of processes
- `option`: 0 for random array, 1 for sorted, and 2 for reverse sorted (not fully implmeneted for the CUDA version just yet)

#### Adapted From 


* Source Code: [https://github.com/ym720/p_radix_sort_mpi/tree/master/p_radix_sort_mpi](https://github.com/54kevinalbert/gpu-mergesort)
 * Author: Kevin Albert


## Bitonic Sort

### MPI

#### Summary

The input gets generated using a for loop that picks random numbers between 0 and 99. The data is evenly distributed across all the processers, then a bitonic sequence is constructed, followed by the processes communicating with each other to exchange data until it is sorted.

The input is 
#### Running

1. Run `. build.sh`

2. Run sbatch:

```
sbatch mpi.grace_job <p> <n> 
```

- `p`: number of processes
- `n`: length of array you want to sort

#### Adapted From 


* Source Code: https://github.com/adrianlee/mpi-bitonic-sort/tree/master

 * Author: Adrian Lee

### CUDA

#### Summary

The CUDA implementation is different in that it goes through the CUDA kernel, and is called with the array of integers, and each GPU thread getting a unique id. The bitonic_sort function then does the same alternating and swapping as above till sorted. 

#### Running

1. Run `. build.sh`

2. Run sbatch:

```
sbatch mergesort.grace_job <t> <n> 
```

- `t`: number of threads
- `n`: how many numbers you want to sort 


#### Adapted From 


* Source Code: http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm

 * Author: Adapted from Lab 3

## Quicksort

### MPI
#### Summary
The input gets generated sequentially, using a for loop to generate `num_of_elements` amount of random numbers. The array is scattered across the different processors and then sorted on each processor using the serial quicksort algorithm. The processors for a binary tree structure where the right child will send its data to the left child to be merged into one array. This process is repeated until the entire array is merged and sorted. Once the array is sorted, the root processor will check if it is sorted correctly and print out the result.

#### Running  
1. Run `. build.sh`
2. Run sbatch:
```
sbatch quicksort.grace_job <n> <t>
```
- `n`: length of array you want to sort
- `t`: number of threads

#### Adapted From
* Source Code: https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/
* Author: GeeksforGeeks (Ashutosh Soni)

### CUDA
#### Summary
The input gets generated sequentially again, also using a for loop to generate `size` elements. As far as I could understand, we split the array into blocks of size `block_size`, with `cThreadsPerBlock` threads in them (128 threads per block) and then sort each block using the kernel function quicksort algorithm. So each block is sorted independently of the others. We finally merge it all together in the same while loop.

#### Running
1. Run `. build.sh`
2. Run sbatch:
```
sbatch quicksort.grace_job <n> <p>
```
- `n`: length of array you want to sort
- `p`: number of processes

#### Adapted From
* Source Code: https://github.com/saigowri/CUDA/blob/master/quicksort.cu
* Author: Sai Gowri

# Performance Evaluation & Report


## Algorithm Comparisons

### Strong Scaling

The below table represents the time consumed communicating between the different threads given a set of random data


| Number of Values | Plot |
|------------------|------|
| 2^16             | <img src="PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_11_0.png" width="400"> |
| 2^18             | <img src="PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_11_1.png" width="400"> |
| 2^20             | <img src="PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_11_2.png" width="400"> |
| 2^22             | <img src="PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_11_3.png" width="400"> |
| 2^24             | <img src="PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_11_4.png" width="400"> |
| 2^26             | <img src="PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_11_5.png" width="400"> |
| 2^28             | <img src="PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_11_6.png" width="400"> |


We observe that overall, the worst performer is MergeCUDA in regards to communication. This is then followed by QuicksortMPI and MergeMPI. The fastest performer is consistenlty MergeCUDA and BitonicCUDA. It is interesting that the MergeCUDA is so poor, indicating that it communicates a lot more throughout its code runtime than any other algorithm.

### Weak Scaling

The table below compares the execution time of different algorithms performance as workload and resources scale proportionally. Ideally, the graph should be a straight horizontal line. 

| Implementation | Plot |
|------------------|------|
| MPI            | <img src="https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/c527a236-691b-48c3-a4ec-c2f017462406" width="400"> |
| CUDA             | <img src="https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/f5a633c2-5cf6-4306-8769-ac4cdc959e36" width="400"> |

We observe that MergeMPIRandom is the worst performer in regards to weak scaling. This is then followed by RadixMPI, BitonicMPI, and QuicksortMPI. Interestingly, although RadixMPIRandom is the 2nd worst performer for weak scaling at less processes, we see that when we increase the number of processes, it becomes a better weak scaler than all the other algorithms.


### Speedup

**Speedup**: Execution time with p processors / execution time with 1 processor

The tables below compares speedup of communication between processors / threads against the number of processors / threads used for varying amounts of input. Ideally, we should a linear upward trend

**The y-axis for these plots should say Speedup, not Time(s). By the time we caught this error**


| Number of Values | Plot |
|------------------|------|
| 2^16             | ![png](PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_12_0.png) |
| 2^18             | ![png](PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_12_1.png) |
| 2^20             | ![png](PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_12_2.png) |
| 2^22             | ![png](PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_12_3.png) |
| 2^24             | ![png](PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_12_4.png) |
| 2^26             | ![png](PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_12_5.png) |
| 2^28             | ![png](PerformanceEval/Plots/AlgorithmComparisons_files/AlgorithmComparisons_12_6.png) |

With speedup as it relates to communication between processors / threads, we noticed that for lower input sizes, the speedup was much lower, and as processor count increased, communication speedup decreased for MPI, and stayed flat for CUDA. 

This is likely due to unnecessary communications creating an overhead when there's too many processors for such a small problem size. As the input size increases, speedup slowly flattens, then trends upwards, which can likely be explained by communications being useful with higher input sizes, increasing processor utilization. CUDA likely remains flat across all input sizes due to GPUs being highly optimized for parallel workloads at a hardware level. 

## Radix Sort Analysis and PerformanceEval/Plots

### MPI

#### Strong Scaling

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/65bd3abb-765c-40b5-ada6-386aae98d0b3)

We see that for Radix Sort, the Comparisons Strong Scale rather well. We do note that there are diminishing returns. Ideally, the graph should be a straight line down, as the number of processes are in log2 scale. We observe that from 2 to 4 processes gives us the most advantage as the slope is the steepest, indicating a pronounced acceleration in efficiency.

| RadixMPI Random | Strong Scaling MPI Reverse |
| --------------- | --------------- |
| ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/9fecf0ed-7927-4cdd-879d-578aee3b8a8d)    | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/56211e51-bed2-4fc3-8f0f-7ba17388306e) |
| ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/20540a19-46b0-4ccb-a14e-a8c35676ad5e) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/5994e730-58de-49f0-9f08-9b03aa71c106) |


One interesting observation to note within strong scaling is the difference between the sorting of random values versus reverse values. Observe the first row, where we see that given an input of random values, the algorithm scales relatively well, especially for higher inputs. However, with an input of reverse input there is virtually no strong scaling observed.


Upon further investigation, we see that the culprit is the communication overhead. We see on the left with random input that as we increase the number of processes, the time trends downward. With reverse input, we observe that on increasing the number of processors, the time actually increases.
The reason for this phenomenon lies in the inner workings of my MPI Radix Sort. Within the code, each process is given a section of the array to sort. After each pass, where the algorithm sorts by 8 bits at a time, the processes communicate amongst each other to determine where in the array their numbers belong. If the process realizes that the value it "owns" belongs in another section of the array, it will communicate as such, and send it off to the process it belongs to. 

With reverse sorted output, we know that the values that a process "owns" will all have to be sent to another process. This results in each process having to both send and receive the entirety array that it owns. Thus, it is no surprise that with a reverse sorted input, the more processes we add, the more time it takes to communicate between processes, and the worse the overall runtime.

Below are the rest of the strong scaling files

**Labels**: 
- Main: The overall execution time
- Communication (Comm): The time taken to communicate between processors
- Comparisons (Comp): The time taken to compare the values as you sort 

| Number of Values | Main | Communication | Comparison |
|------------------|------|---------------|------------|
| 2^16             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ab9a1f24-256c-4b3b-8bb4-5c5947df38d5) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b05dd462-35fc-4ce9-9f95-7900dc5dbcac) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/20b5f851-2906-484b-b5aa-39839e41db90) |
| 2^18             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/feee5dd3-7ad1-4a06-bf55-50d7780f3cc7) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/aee4e9d8-972c-4982-a1c1-7157f5055a49) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b7e07a46-ca44-4bb0-b47a-ef33809bc871) |
| 2^20             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ae11bc54-e6dc-49d4-b4a9-27451108614b) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/7a5dcf4f-e1e2-4477-a7dc-d7ff77904c30) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/5f39e6b5-c9dc-4190-9885-c36ec3aae964) |
| 2^22             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/754a4192-8d3a-4d7a-bac9-01d8bbcc644e) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b76fd6c2-658d-49f7-afe3-b8d47c412152) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b0c591cb-3900-489e-afbe-0f615a249e7f) |
| 2^24             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/3c3849c4-46e5-4afe-b99a-064e7870bdb5) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/366fcabb-84d1-43ec-9da6-032e13a19af4) |  |
| 2^26             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/f160ee3a-4cd6-4ffb-832a-43bc5b3426a8) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/72474e1a-8366-4e90-bfc0-a1633ff123e4) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/f59dccc6-62c6-4e93-b4e0-cd7f71d07ee9) |
| 2^28             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/c8cef387-1681-4c89-a437-99e2cba36ced) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/a59d8f9f-7769-4d40-9bc2-d1520153cbed) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/39a32a23-0c5d-4f76-9de5-7ee8ce5d1618) |



#### Weak Scaling

| Comp Large | Comm |
| --------------- | --------------- |
| ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/3aeb583f-a3a9-45fe-a89f-84f9160e2ff0) |  ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/43a3c104-4adf-419a-923c-31966304bd66) |

   With Random Input, we observe that both comp_large and comm weak scale quite well. Although the lines seem to trend upwards, we see that the y axis scale is quite small, and so we may consider both of these trends as practically straight lines, with comp large weak scaling the best.

Below are the remaining Weak Scaling files


**Labels**: 
- Main: The overall execution time
- Communication (Comm): The time taken to communicate between processors
- Comparisons (Comp): The time taken to compare the values as you sort 

#### Main

| Input Type | Main | Comm | Comp Large |
|------------|------|------|------------|
| Random     | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ee7b4b82-6df8-4b7c-9be9-7300fee5eda2) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/fbdca2b8-9547-42f7-bc41-ebed4668c88e) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/c938646c-bd2b-423b-b2ee-ff421761d059) |
| Sorted     | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b0145d46-66ed-40da-a360-254b174db7f8) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/371dd889-6c58-4744-ab58-efed79158e98) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/200a7d75-19c3-4480-863e-1fe39455109b) |
| Reversed   | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/2d0cdacc-34a3-413a-9205-f18f05184c74) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/4717d98b-a5a4-40b3-a2c7-305c4e21978b) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/11cda8ef-89d1-4395-8d68-c1aa97432096) |
| Perturbed  | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/57280eb1-eff8-4cdd-8a55-52ea1a2dbc6d) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/544df122-4d32-42c9-a8cb-3b7c086688d6) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/8690a6cf-6f94-4522-9adc-694c15d2ecf7) |


#### Speedup

| Comp Large | Comm |
| --------------- | --------------- |
| ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/e672d180-6960-4234-8178-fb0989e3223d) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/e4b578b1-a5f5-405b-ba64-84fe85e15877) |

The speedup for comp_large is relatively decent. That is, we do see an upward trend, as we increase the number of processes. However, the speedup is not ideal by any means. We see that the highest speedup is achieved with an input of 2^22, where at 2^8 processors, we have a speedup of 120. Ideally, 2^8 processors would give us a speedup of 2^7. This is not so. One possible explanation is that as we increase the number of processors, there is just more overhead. Right before every processor prepares to receive values from other processors, each individual processor has to determine if it has enough space to receive x number of values. Thus, with more processors, more communication follows, meaning more computation as each processor takes more time to compute the size of the array it should be prepared to receive.

We also observe that communication has extremely poor speedup. In fact, we see a downward trend. This makes sense as we increase the number of processors, we increase the communication time, leading to poor and even negative speedup. 

Below are the remaining Speedup Plots

| Input Type | Main | Comm | Comp Large |
|------------|------|------|------------|
| Random     | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/cde53e79-1465-4cfc-823d-0503dd3440a1) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/42212d63-26ac-4fba-ab78-330c05143286) |  |
| Sorted     | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/f0008580-cbef-4980-9f9a-9a29ebd0f1b1) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/8ad46a2e-fc02-4fa4-82b0-1b28441fd7d5) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/f03ac0ea-c03a-4c7c-9b10-187e51833eb0) |
| Reversed   |  | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/4f6ca8df-0283-497b-b223-28b26b24b4f8) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/cab9b7b1-a1e0-404a-873f-d1a78c2f3fbd) |
| Perturbed  | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/c0afcae7-3ab3-4058-9a23-c5c01ee25045) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/118b065d-bbfd-473b-b69e-76331a8f762b) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/bd8c463e-9341-43a6-98fd-235dcda99acf) |


### CUDA

#### Strong Scaling

| All Input Types | Random Input Type |
| --------------- | --------------- |
|![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/698235a6-0bae-45d7-8c79-9dd70e8e333f) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/e3aa3d62-148a-4c6f-a0cc-d3d572c82070) |


 We observe the table on the left. We see that from worst to best performance based on input type is as follows: random, perturbed, sorted, reverse. Most surprising is the fact that reverse performed the best. I believe this may be due to the block shuffle that occurs. After each thread appropriately sorts its own block, there is a function that shuffles the block to its correct space. As the sorting algorithm looks at only 2 bits at a time, in a sorted input, a lot more shuffling may occur. 1, 2, … 10, 11, 12, … 20, … 40 would shuffle first to 1, 11, 21, … 2, 12, 22, … which involves much more unnecessary shuffling than if the input is in reverse order. In reverse order, although there is shuffling, the shuffling brings you closer to the sorted output, versus in sorted order, the shuffling brings you further away from sorted order. 

By observing just an input of 2^28, we see that after 2^7 threads, the overhead becomes too much, and we see a diminishing returns trend.

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/9ed5e6dd-4cea-4f51-959e-84c8235b445a)

This exact same trend can be observed through comp_large. After 2^7 processors, the overhead becomes too large to give much benefit.

Below are the remaining Strong Scaling plots



| Number of Values | Main | Communication | Comparison |
|------------------|------|---------------|------------|
| 2^16             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/a78da7b0-a2b7-4296-a5db-34c31a885847) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b0664bad-4410-4f79-917f-f51fb8b77891) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/721df98a-93ab-494d-909e-1becdd38b0a0) |
| 2^18             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/578052f5-d8c4-4c52-a383-ff9d6dde6735) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/7aa4c301-0128-43df-9ec0-c61fc0e1ab9e) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/2af7ea5e-38dd-4b55-b616-50c1e445a338) |
| 2^20             |  | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/f461388f-6ec1-4c8a-84bf-9b968b8b893e) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/37ca5b3d-857d-4887-865f-4ba1eae282ec) |
| 2^22             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/3247a9ec-4f86-48b4-be8c-817bc21c381f) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ad70d1af-badb-4ca4-b636-40ec97ef9815) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ee10595e-e848-4771-9eb7-5a94b004bf16) |
| 2^24             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/246644a0-4e94-4d43-8b36-b8dbbb4c4aa2) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/f8d83e89-b14a-49e9-a9c4-a8e288edb716) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/f016af68-2d8e-410d-a852-a36d36a565fd) |
| 2^26             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/cf4ad9d3-c47c-40ef-ba5a-93510826d94a) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/55fb2fa3-78fa-4f92-b393-edd70d25e651) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/1b76df98-e372-4f34-abfe-e4b9b9e0dedd) |
| 2^28             | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/d07583fb-625d-490f-96a8-29107d364b6a) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/a052bef2-957b-4f63-b159-2d2fe219131d) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/e56e7fdd-f9ed-482d-a755-50da2f6358fa) |


#### Weak Scaling

| Comp Large | Comm |
| --------------- | --------------- |
| ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/4e1e40ea-0ed7-4435-9b0c-cb41f983aa8d) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/32a6ad14-a783-4428-a419-2216843d9ba3) |


The CUDA Radix Sort implementation weak scales only moderately well, with the difference between 2^6 threads and 2^10 threads only benign about 0.5 for both comp_large and comm. This again, is due to all the shuffling needed between blocks as the number of threads increases.

Below is the remaining plots for Weak scaling

| Input Type | Main | Comm | Comp Large |
|------------|------|------|------------|
| Random     | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/5c578d2b-af00-4ce8-825e-ef035dfde7f5) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/39541da1-c9bc-4647-a6bf-ec7913ccf745) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/0862fc6d-bd66-4f53-9e33-6f941e2ca629) |
| Sorted     | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b96056d7-4f05-43a4-84b6-0d194fa7e6e9) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/7908f0c8-d527-4e8b-b78f-51d165e53f8e) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/0ed29767-20a3-459e-8bc9-e434419b8ac1) |
| Reversed   | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ddf9ca74-7d73-4c29-ad6e-ca781380b51a) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/3d8faf55-c785-4863-bf70-41f8a2f94246) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/91f6e157-556c-4363-abe5-58410e3bdc74) |
| Perturbed  | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/7239dc20-de78-4316-961b-069bfbf179a6) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/7cd119dd-85db-46c3-a69a-0b23d011eeaa) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/24c9616f-85a1-4da4-95f9-af75b5890af9) |


#### Speedup

| Comp Large | Comm | Main |
| --------------- | --------------- | --------------- |
| ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/7d38b1c7-e6d8-4222-a65f-f8cc202fefe8) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/92eaf742-ae49-4e7a-850d-347ead196b96) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/bdfe0d4f-73c3-41fe-add8-447c3f4c06c2) |


In the speedup, we see the same trend observed with strong scaling. That is, that the best speedup occurs from 2^6 to 2^7, followed by only diminishing returns. From 2^6 to 2^7 the speedup is only moderately good for comp_large, with the speedup resulting in 1.3, which is close to 2.The comm shows poor speedup, and the the main function demonstrates moderate speedup for 2^6 to 2^7 processors. Main is a poor measure as it includes data correction and input generation. Poor speedup in comm makes sense as the time to communicate will only increase rather than decrease as we add more threads.

Below are the remaining plots for Speedup


| Input Type | Main | Comm | Comp Large |
|------------|------|------|------------|
| Random     | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b0f6f832-3da9-45f6-ac03-45455fecfdc2) |  | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/3de25304-9100-4a17-b6df-ecc81214a21d) |
| Sorted     | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/951e8dc4-3777-49f9-8a90-77f7b1882ba7) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/e9e2b341-60b8-4496-afcd-790578c33d02) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/cd5148b3-fa16-4d45-baee-17b18f0b404b) |
| Reversed   | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/34c5c688-d8aa-4369-a235-5f509a6d5fc5) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/432dd49d-5357-403b-90b0-902bf23f5a8d) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ed3ab1df-e115-4c7b-a08b-8eb2f6372514) |
| Perturbed  | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/2ec15469-695f-4f4b-8ef2-15e3366a773f) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ec6381c0-fcde-4d0e-bad8-ac91feb79565) | ![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/3bdf477b-ac6d-4303-adc8-55480c76a93b) |

#### Main

Random

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/b0f6f832-3da9-45f6-ac03-45455fecfdc2)

Sorted

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/951e8dc4-3777-49f9-8a90-77f7b1882ba7)

Reversed

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/34c5c688-d8aa-4369-a235-5f509a6d5fc5)

Perturbed

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/2ec15469-695f-4f4b-8ef2-15e3366a773f)


#### Comm

Sorted

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/e9e2b341-60b8-4496-afcd-790578c33d02)

Reversed

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/432dd49d-5357-403b-90b0-902bf23f5a8d)

Perturbed

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ec6381c0-fcde-4d0e-bad8-ac91feb79565)

#### Comp Large 

Random

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/3de25304-9100-4a17-b6df-ecc81214a21d)

Sorted

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/cd5148b3-fa16-4d45-baee-17b18f0b404b)

Reversed

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/ed3ab1df-e115-4c7b-a08b-8eb2f6372514)

Perturbed

![image](https://github.com/Sofia-Ortega/ParallelComputingProject/assets/40405324/3bdf477b-ac6d-4303-adc8-55480c76a93b)







## Mergesort Analysis and PerformanceEval/Plots

**We realized too late that the Merge Sort Plots have transparent axis backgrounds - if you're unable to view it, please try viewing the plots by turning on light mode in your browser or IDE, as this should resolve the problem. We apologize for any inconvenience this may cause.**

### Strong Scaling

I used the strong scaling PerformanceEval/Plots for the input size of 2^28 for my analysis. I wanted to look at the largest input size to get the best understanding of how well the algorithm scales.

#### MPI

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comm_Strong_Scaling:_Input_Size_1048576.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comm_Strong_Scaling:_Input_Size_16777216.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comm_Strong_Scaling:_Input_Size_262144.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comm_Strong_Scaling:_Input_Size_268435456.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comm_Strong_Scaling:_Input_Size_4194304.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comm_Strong_Scaling:_Input_Size_65536.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comm_Strong_Scaling:_Input_Size_67108864.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_1048576.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_16777216.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_262144.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_268435456.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_4194304.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_65536.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_67108864.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Main_Strong_Scaling:_Input_Size_1048576.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Main_Strong_Scaling:_Input_Size_16777216.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Main_Strong_Scaling:_Input_Size_262144.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Main_Strong_Scaling:_Input_Size_268435456.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Main_Strong_Scaling:_Input_Size_4194304.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Main_Strong_Scaling:_Input_Size_65536.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/StrongScaling/Main_Strong_Scaling:_Input_Size_67108864.png)

By looking at the PerformanceEval/Plots for the comp_large region, we can see that there is certainly a decrease in time as the number of processors increases. Interestingly, the communication time seems to be increasing with the number of processors, but it then drops back down past 2^5 processors. I did notice that the communication overhead is the dominating factor in the runtime as the input size gets large and the number of processors gets large as well. Overall, the MPI mergesort algorithm seems to strong scale very well.

#### CUDA

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comm_Strong_Scaling:_Input_Size_1048576.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comm_Strong_Scaling:_Input_Size_16777216.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comm_Strong_Scaling:_Input_Size_262144.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comm_Strong_Scaling:_Input_Size_268435456.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comm_Strong_Scaling:_Input_Size_4194304.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comm_Strong_Scaling:_Input_Size_65536.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comm_Strong_Scaling:_Input_Size_67108864.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_1048576.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_16777216.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_262144.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_268435456.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_4194304.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_65536.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Comp_Large_Strong_Scaling:_Input_Size_67108864.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Main_Strong_Scaling:_Input_Size_1048576.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Main_Strong_Scaling:_Input_Size_16777216.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Main_Strong_Scaling:_Input_Size_262144.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Main_Strong_Scaling:_Input_Size_268435456.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Main_Strong_Scaling:_Input_Size_4194304.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Main_Strong_Scaling:_Input_Size_65536.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/StrongScaling/Main_Strong_Scaling:_Input_Size_67108864.png)

My CUDA mergesort algorithm seems to be quite interesting. It seems to be running suspicously fast; on the largest input size, it seems to take a fraction of a second to sort everyting. All of my runs did pass the correctness check, but I still not sure if my code is working quite right. It just seems too fast to be true. With this very fast runtime, it does not seem to strong scale since it is already so fast. The runtime of my CUDA implementation is pretty much all communication between the GPU and the CPU. It is interesting to note that the communication time seems to decrease with the number of threads; I am not sure what would cause this kind of trend.

### Strong Scaling Speedup

I used the PerformanceEval/Plots for 1% perturbed for my analysis below. This was not for any particular reason as all the input types showed similar trends.

#### MPI

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Comm_Speedup_MergeMPIPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Comm_Speedup_MergeMPIRandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Comm_Speedup_MergeMPIReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Comm_Speedup_MergeMPISorted.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Comp_Large_Speedup_MergeMPIPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Comp_Large_Speedup_MergeMPIRandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Comp_Large_Speedup_MergeMPIReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Comp_Large_Speedup_MergeMPISorted.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Main_Speedup_MergeMPIPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Main_Speedup_MergeMPIRandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Main_Speedup_MergeMPIReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/Speedup/Main_Speedup_MergeMPISorted.png)

The MPI mergesort algorithm shows great speedup for the comp_large region. It actually seems that the computation still has lots of room to grow as it hasn't started plateauing; however, the speedup for the main region does show that the algorithm has reached its highest speedup of about 4.5 for the largest input size. This is because the main region includes input generation, computation, and correctness check. The correctness check and input generation are not non-parallel regions which causes the speedup to flatten out.

#### CUDA

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Comm_Speedup_MergeCUDAPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Comm_Speedup_MergeCUDARandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Comm_Speedup_MergeCUDAReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Comm_Speedup_MergeCUDASorted.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Comp_Large_Speedup_MergeCUDAPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Comp_Large_Speedup_MergeCUDARandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Comp_Large_Speedup_MergeCUDAReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Comp_Large_Speedup_MergeCUDASorted.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Main_Speedup_MergeCUDAPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Main_Speedup_MergeCUDARandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Main_Speedup_MergeCUDAReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/Speedup/Main_Speedup_MergeCUDASorted.png)

The speedup for the CUDA version does not look very good. It hardly gets above 1 at some thread counts, but I don't think that this is the whole story. Since the algorithm is taking a fraction of a second already, it doesn't show any speedup. However, this doesn't really matter since it is already incredibly fast. I wonder if running on larger input sizes would cause it to take longer and show more of a speedup, but with the data I collected there isn't much to look at.

### Weak Scaling

For weak scaling, I again looked at the perturbed data PerformanceEval/Plots. This was not for any particular reason as all the input types showed similar trends.

#### MPI

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Comm_Weak_Scaling_MergeMPIPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Comm_Weak_Scaling_MergeMPIRandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Comm_Weak_Scaling_MergeMPIReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Comm_Weak_Scaling_MergeMPISorted.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Comp_Large_Weak_Scaling_MergeMPIPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Comp_Large_Weak_Scaling_MergeMPIRandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Comp_Large_Weak_Scaling_MergeMPIReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Comp_Large_Weak_Scaling_MergeMPISorted.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Main_Weak_Scaling_MergeMPIPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Main_Weak_Scaling_MergeMPIRandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Main_Weak_Scaling_MergeMPIReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/MPI/WeakScaling/Main_Weak_Scaling_MergeMPISorted.png)

The computation region weak scales very well as the plot is almost a perfect straight line. However, the communication and main regions do not weak scale at all. The main region shouldn't weak scale since it has to do more comparisons when running the correctness check and generates more data. It also makes sense that the communication region does not weak scale because it the amount of communication is directly tied to the size of the problem.

#### CUDA

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Comm_Weak_Scaling_MergeCUDAPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Comm_Weak_Scaling_MergeCUDARandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Comm_Weak_Scaling_MergeCUDAReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Comm_Weak_Scaling_MergeCUDASorted.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Comp_Large_Weak_Scaling_MergeCUDAPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Comp_Large_Weak_Scaling_MergeCUDARandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Comp_Large_Weak_Scaling_MergeCUDAReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Comp_Large_Weak_Scaling_MergeCUDASorted.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Main_Weak_Scaling_MergeCUDAPerturbed.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Main_Weak_Scaling_MergeCUDARandom.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Main_Weak_Scaling_MergeCUDAReverse.png)

![png](PerformanceEval/Plots/MergeSortPlotting_files/CUDA/WeakScaling/Main_Weak_Scaling_MergeCUDASorted.png)

For CUDA, the computation region weak scales very well but the main and comm regions do not. The main region acts the same as in MPI, and the amount of data being sent to/from the GPU increases with the problem size so it wouldn't weak scale either.

## Bitonic Sort PerformanceEval/Plots

This section will have all PerformanceEval/Plots for our Bitonic Sorting Algorithm, including both MPI and CUDA implementations for Random, Sorted, Reversed, and 1% Perturbed input types. The PerformanceEval/Plots are grouped by strong scaling, speedup, and weak scaling for Main, Comm, and Comp_Large Caliper regions.  

### Strong Scaling

#### MPI

For the MPI implementation, we see relatively the same trends across all input types. The only notable exception is random input type, which has a much higher time taken for smaller processor numbers. This can likely be explained by sorted, reversed, and 1% perturbed having a predictable trend which means swapping becomes easy - it's either already sorted or it just needs to be switched to the opposite side of the array, or only 1% of the array needs to be actually sorted. With random, however, there's no trend, and there's likely much more overhead that's created as a result of all the swapping and comparisons that happen.


On the lowest input sizes, we see a near exponential growth pattern in time taken as processor count increases, which can mostly be attributed to communication overheads caused by bringing a metaphorical gun to a knife fight - with too many processors and too little computations that need to be done, the waste communication time of sending empty work to a processor outweighed the benefits of parallelization.


On the largest input sizes, we see a near exponential decline in time taken as processor count increases, with increases in time taken at the highest processor count. This is likely because the input size was so large that there was meaningful work for all the processors to carry out meaning that the benefits of parallelization weren't eaten up by communication overheads. At the very end, the high processor count likely caused communication overheads to slightly eat into the benefits of parallelization, but with an even larger problem size, this would likely be mitigated and you would see a consistent decline in time taken with increasing processor count.

The comm times gradually increasing as input size increases corroborates this, as does the consistent, sharp decline of comp_large.  
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_0.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_1.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_2.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_3.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_4.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_5.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_6.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_7.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_8.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_9.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_10.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_11.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_12.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_13.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_14.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_15.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_16.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_17.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_18.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_19.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_12_20.png)
    


#### CUDA

For CUDA, all input types behaved the same (mostly the same with negligible differences) across all input sizes, with no discernable trend until the input size of 67108864, at which point we see compute times measurably decrease on the main function across all input types. Both the comp_large and comm regions stay at very similar times throughout all input sizes, even when there did start to be a measurable decrease in time at the larger input sizes. This is likely due to GPUs being designed for high throughput with thousands of cores, data locality, and other facts. The scale should be closely examined for all CUDA PerformanceEval/Plots. It's also important to note that the runtime's biggest differential across all input types and sizes is just one second, from ~2.2 seconds to ~1.2 seconds at the largest inpiut size, showing the impressive benefits of using a GPU for complex computations like this. 

    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_0.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_1.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_2.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_3.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_4.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_5.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_6.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_7.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_8.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_9.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_10.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_11.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_12.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_13.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_14.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_15.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_16.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_17.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_18.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_19.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_14_20.png)
    


### Speedup

#### MPI

For MPI, we see that the main function's speedup has the same trend across all input types. However, the scale of the speedup changes dramatically with input type. Random input experienced the highest speedup at the higher processor counts, achieving up to around 9, with a drop off at 1024 processors. This is likely due to there being a much higher amount of computations that are needed ot sort a random array, which will benefit from more processors way more than mostly pattern sorted arrays like 1% perturbed, sorted, and reverse input types.  We also see that for smaller input sizes, the speedup is actually negative as processor numbers increase. The decrease in speedup at the highest levels for smaller input sizes, and at 1024 for the largest input size, is likely caused by increases in communication overheads, which stays constant in the begining despite exponentially increasing processor counts. Comp_Large exponentially increases across all input types, which further corroborates that slowdowns were likely caused by communication overheads. 
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_0.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_1.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_2.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_3.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_4.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_5.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_6.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_7.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_8.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_9.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_10.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_18_11.png)
    


#### CUDA

Our CUDA implementation of bitonic sort had a much smaller differential than MPI, with a max speedup of 2. This can likely be explained by GPUs' nature being much more focused on parallelism, data locality, and much higher throughout. We don't believe the low speedup is due to the algorithm, as there is some speedup, and the comp_large and comm times stay largely flat across all input sizes. 
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_0.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_1.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_2.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_3.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_4.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_5.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_6.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_7.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_8.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_9.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_10.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_20_11.png)
    


### Weak Scaling


#### MPI

For the MPI implementation on a weak scaling front, we see nearly identical, strong performance across all input types and processor numbers. All input types start at around a time of .5 and end at around 3.5 for the same problem size with more processors. Comm and Comp_Large weak scaling measures remain negligible, but with an upward trend, across all input types. 
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_0.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_1.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_2.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_3.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_4.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_5.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_6.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_7.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_8.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_9.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_10.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_24_11.png)
    


#### CUDA

Our CUDA implementation had weaker weak scaling than our MPI implementation, which can likely be explained by the high levels of parallelism that GPUs have already, corroborated by very negligible comm and comp_large times across all input types.
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_0.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_1.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_2.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_3.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_4.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_5.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_6.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_7.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_8.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_9.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_10.png)
    
![png](PerformanceEval/Plots/BitonicSortPlotting_files/BitonicSortPlotting_26_11.png)
    

## Quick Sort PerformanceEval/Plots

### Overall: 
There are a few disclaimers. My CUDA implementation was not able to achieve the problem sizes that we were initially given. As a result, I used problem sizes from $2^{10}$ to $2^{16}$, increasing in increments of powers of 2. For my MPI implementation, however, I was able to get everything working, so I used the problem sizes from $2^{16}$ to $2^{28}$, increasing in increments of powers of 2.


### Strong Scaling

#### Analysis:
**Strong scaling** is when you increase the number of processses, but keep the problem size the same. As you can see, this MPI implementation scales pretty well across all of the different kinds of inputs. Between problem sizes $2^{16}$ and $2^{20}$, you can see that time taken to run the entire algorithm actually increases with process size increase. It is only after $2^{22}$ that it actually decreases. This is due to the fact that it takes a lot more time to communicate between the higher process sizes and there is a lot of organizational overhead. It is only on large input sizes that the communication time is dwarfed by the actual computation time, so increased process size actually helps. The same behavior can be seen in the graphs measuring communication. The same explanation can be used for these graphs. The communication time is actually increasing with process size until the input size is large enough that the communication time is dwarfed by the computation time. However, something peculiar is seen in Comm. On random input types with large problem sizes, there is an inverted graph that is logarithmic in nature compared to the other graphs. I think that for smaller process counts in random inputs with large problem sizes, it takes a long time to communicate and merge chunks of data on small process sizes during the ```MPI_Send``` and ```MPI_Recv``` calls. However, as the process size increases, the communication time decreases. We can support these arguments by looking at the computation graphs, in which the computation time decreases with process size increase, which makes sense since without taking in communication time, the computation time should decrease with process size increase exponentially.

#### MPI
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_0.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_1.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_2.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_3.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_4.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_5.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_6.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_7.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_8.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_9.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_10.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_11.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_12.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_13.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_14.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_15.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_16.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_17.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_18.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_19.png)
    



    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_10_20.png)
    

### CUDA

#### Analysis:
There are a few observations to make with this CUDA implementation. It takes much longer time on all thread counts to run the algorithm for reverse sorted input types, as well as communication and computation. This is because the pivot is always chosen to be the first element in the array, so if the array is reverse sorted, then the pivot will always be the largest element in the array. This means that the array will be split into two subarrays of size 1 and size n-1, which is the worst case for quicksort. This is why the reverse sorted input types take much longer to run. The random input types take the least amount of time to run, which makes sense since the pivot will be chosen randomly on random input, so the array will be split into two subarrays of size n/2 and n/2, which is the best case for quicksort. There are hills and valleys in communication times for all input times, with sorted input doing the worst in terms of communication. Brute computation seems to be pretty constant across all input types, however, I am not sure if this is a product of my earlier disclaimer that I was not able to get the CUDA implementation to work for large problem sizes without timing out.
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_0.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_1.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_2.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_3.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_4.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_5.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_6.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_7.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_8.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_9.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_10.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_12_11.png)
    


### Strong Scaling Speedup

#### MPI

#### Analysis:
When measing **speedup**, we are measuring the improvement in parallelization versus time taken sequentially. For large input sizes in all input types for total runtime, we see a similar trend in speedup. There is an almost parabolic route with $2^{28}$, the largest input size, where there is the highest speedup. However, with smaller input sizes, we actually see a gradual decrease in speedup due to process organizational overhead. In brute computation time, we see exponentially increasing speedup as process count increases, which makes sense since as process count increases, computation time decreases. With communication, we see similar graphs to main in input types of 1% perturbed. However, for the other three input types we see a gradual decrease in speedup as process count increases, which makes sense because of the organizational overhead mentioned before.

    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_0.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_1.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_2.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_3.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_4.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_5.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_6.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_7.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_8.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_9.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_10.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_16_11.png)
    


#### CUDA


#### Analysis:
For the CUDA implementation, we actually see a few polar opposites from the MPI implementation. First of all, the speedup gradually decreases for all input types for entire runtime. This is because the CUDA implementation was not able to achieve the same problem sizes as the MPI implementation without taking a long time. For brute computation on random input types, speedup becomes 0 on any process count higher than 64. I believe that this is also because of the problem size limitation. For communication, the trends are almost identical to the graphs for the entire algorithm runtime. This is of organizational overhad, as mentioned before.


![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_0.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_1.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_2.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_3.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_4.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_5.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_6.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_7.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_8.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_9.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_10.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_18_11.png)
    


### Weak Scaling




#### MPI
#### Analysis:
When talking about **weak scaling**, we are talking about increasing the problem size and the process size proportionally. For the MPI implementation, we see that the total runtime increases exponentially with problem size increase when testing full algorithmic runtimes. Something to note is that on Random input types, the scale goes from 1-8 seconds, however for the other 3 input types, it goes from 0.5 to 3.0-3.5 seconds. With graphs measuring communication, we see a similar trend as well as similar scale differences. However, with the brute computation, we see a parabolic route. This is because until a certain point, the problem size is too large for the proportional increase in process size. There is a saddle point, however, where the problem size is large enough that the process size increase is proportional and we start to see a decrease in time. With 1% Perturbed, we see a linear decrease in time as problem size and process count increases. However, the scale is so small that we can consider these almost constant lines.
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_0.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_1.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_2.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_3.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_4.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_5.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_6.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_7.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_8.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_9.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_10.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_22_11.png)
    


#### CUDA

#### Analysis:
With CUDA implementation, we see almost identical trends as the MPI implementation. However, the scale is much, much larger. This is because the CUDA implementation was not able to achieve the same problem sizes as the MPI implementation without timing out. We see the same trends as before though. We see a linear increase in time taken in both full algorithm runtime as well as communication. We do see a parabolic graph in brute computational time, but it is an inverted parabola. For Random and Reverse-Sorted. However, with Sorted and 1% Perturbed, we see a linear increase and a somewhat zig-zag shape. However, due to the difference in scale being so small, we can consider these constant lines as well.

![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_0.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_1.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_2.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_3.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_4.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_5.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_6.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_7.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_8.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_9.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_10.png)
    
![png](PerformanceEval/Plots/QuickSortPlotting_files/QuickSortPlotting_24_11.png)
    

