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

This section will have all plots for our Bitonic Sorting Algorithm, including both MPI and CUDA implementations for Random, Sorted, Reversed, and 1% Perturbed input types. The plots are grouped by strong scaling, speedup, and weak scaling for Main, Comm, and Comp_Large Caliper regions.  

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

For CUDA, all input types behaved the same (mostly the same with negligible differences) across all input sizes, with no discernable trend until the input size of 67108864, at which point we see compute times measurably decrease on the main function across all input types. Both the comp_large and comm regions stay at very similar times throughout all input sizes, even when there did start to be a measurable decrease in time at the larger input sizes. This is likely due to GPUs being designed for high throughput with thousands of cores, data locality, and other facts. The scale should be closely examined for all CUDA plots. It's also important to note that the runtime's biggest differential across all input types and sizes is just one second, from ~2.2 seconds to ~1.2 seconds at the largest inpiut size, showing the impressive benefits of using a GPU for complex computations like this. 

    
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
    

