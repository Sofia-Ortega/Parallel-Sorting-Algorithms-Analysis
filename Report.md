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
  
- Hyper Quick Sort (MPI + CUDA)
  
  Pseudo-code: [Source](https://www.tutorialspoint.com/parallel_algorithm/parallel_algorithm_sorting.htm)

```
procedure HYPERQUICKSORT (B, n)
begin

   id := process’s label;
	
   for i := 1 to d do
      begin
      x := pivot;
      partition B into B1 and B2 such that B1 ≤ x < B2;
      if ith bit is 0 then
		
      begin
         send B2 to the process along the ith communication link;
         C := subsequence received along the ith communication link;
         B := B1 U C;
      endif
      
      else
         send B1 to the process along the ith communication link;
         C := subsequence received along the ith communication link;
         B := B2 U C;
         end else
      end for
		
   sort B using sequential quicksort;
	
end HYPERQUICKSORT

```
  
