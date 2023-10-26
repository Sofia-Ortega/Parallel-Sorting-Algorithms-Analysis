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

## 2. Brief project description 

(what algorithms will you be comparing and on what architectures)

- Enumeration Sort (MPI + CUDA)
  
  Pseudo-code: [Source](https://www.tutorialspoint.com/parallel_algorithm/parallel_algorithm_sorting.htm)
  
```
procedure ENUM_SORTING (n)

begin
   for each process P1,j do
      C[j] := 0;
		
   for each process Pi, j do
	
      if (A[i] < A[j]) or A[i] = A[j] and i < j) then
         C[j] := 1;
      else
         C[j] := 0;
			
   for each process P1, j do
      A[C[j]] := A[j];
		
end ENUM_SORTING
```
  
- Odd-Even Transposition Sort (MPI + CUDA)
  
  Pseudo-code: [Source](https://www.tutorialspoint.com/parallel_algorithm/parallel_algorithm_sorting.htm)

```
procedure ODD-EVEN_PAR (n) 

begin 
   id := process's label 
	
   for i := 1 to n do 
   begin 
	
      if i is odd and id is odd then 
         compare-exchange_min(id + 1); 
      else 
         compare-exchange_max(id - 1);
			
      if i is even and id is even then 
         compare-exchange_min(id + 1); 
      else 
         compare-exchange_max(id - 1);
			
   end for
	
end ODD-EVEN_PAR
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
  
