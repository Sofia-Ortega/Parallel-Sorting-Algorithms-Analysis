# CSCE 435 Group project

## 1. Group members:
1. Will Thompson
2. Kirthivel Ramesh
3. Sofia Ortega
4. Sriabhinandan Venkatarama

The team will be communicating primarily through **Discord**. 


---

## 2. Project topic

Sorting Algorithms

### 2. Brief project description 

We will be comparing the performance of the following algorithms with a variety of differing array inputs. These array input will either be sorted, reversed, or randomly selected values. We will also be measuring how well each algorithm scales. We plan to implement each algorithm using MPI to serve the data amongst multiple GPUs that will then use CUDA. Once each part gets sorted on the GPUs, they will be merged either sequentially or in parallel.

- __Parallel Radix Sort (MPI + CUDA)__

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
  
- Odd-Even Transposition Sort (MPI + CUDA)
  
  Pseudo-code: [Source](https://ethz.ch/content/dam/ethz/special-interest/infk/chair-program-method/pm/documents/Verify%20This/challenge3.pdf)

```
process ODD-EVEN-PAR(n, id, myvalue)
 // n … the length of the array to sort
 // id … processors label (0 .. n-1)
 // myvalue … the value in this process
begin
 for i := 0 to n-1 do
 begin
   // alternate between left and right partner
   if i+id is even then
     if id has a right neighbour
       sendToRight(myvalue);
       othervalue = receiveFromRight();
       myvalue = min(myvalue, othervalue);
     else
       if id has a left neighbour
         sendToLeft(myvalue);
         othervalue = receiveFromLeft();
         myvalue = max(myvalue, othervalue);
  end for
end ODD-EVEN-PAR

for i := 0 to array.length-1
 process[i] := new ODD-EVEN-PAR(n, i, array[i])
end for

start processes and wait for them to finish

for i := 0 to array.length-1
 array[i] := process[i].myvalue
end for
```

- Parallel Merge Sort (MPI + CUDA)
  
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
  
- Parallel Quick Sort (MPI + CUDA)
  
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

## Radix Sort


### MPI

#### Summary

Implemented Radix Sort with MPI. 
We initialize the array with `inputgen.cpp` to generate the input in parallel.
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


 # Questions

 1. For Radix sort, I was a bit confused on what was considered a comp_small vs a comp_large. I would love clarification to see if I marked the correct areas correctly.

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


 # Questions

 1. I was also confused on comp_small and comm_small for mergesort. I feel that everything is a large computation and a large communication.
 2. Would you guys have any good resources on how the CUDA verson of mergesort works that goes into the details. I was unable to find any great resources and find the code a little bit confusing.

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

# 4. Performance Evaluation

Please see our pdf file attached. Called **Parallel Computing Plots.pdf**

You may also find it at this [link](https://docs.google.com/document/d/1r1xJd--YJmMYDUgpxQBdYJppri3Gfnrgwj0WtUlYIFU/edit)

## Bitonic Sort Plots


### Strong Scaling

#### MPI

    
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
    


### CUDA


    
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
    

## Quick Sort Plots

### Overrall:


### Strong Scaling

#### MPI

    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_10_0.png)
    

    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_10_2.png)
    

![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_10_4.png)
   

    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_10_6.png)
    

    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_10_8.png)
    

    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_10_10.png)
    

    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_10_12.png)
    

#### CUDA


    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_0.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_1.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_2.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_3.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_4.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_5.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_6.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_7.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_8.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_9.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_10.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_12_11.png)
    


### Strong Scaling Speedup

#### MPI



    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_0.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_1.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_2.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_3.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_4.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_5.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_6.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_7.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_8.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_9.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_10.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_16_11.png)
    


#### CUDA



    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_0.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_1.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_2.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_3.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_4.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_5.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_6.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_7.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_8.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_9.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_10.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_18_11.png)
    


### Weak Scaling




#### MPI

    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_0.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_1.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_2.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_3.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_4.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_5.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_6.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_7.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_8.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_9.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_10.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_22_11.png)
    


#### CUDA

    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_0.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_1.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_2.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_3.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_4.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_5.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_6.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_7.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_8.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_9.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_10.png)
    
![png](PerformanceEval/plots/QuickSortPlotting_files/QuickSortPlotting_24_11.png)
    

