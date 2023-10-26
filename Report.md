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

- Parallel Radix Sort (MPI + CUDA)
  
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
  
- Odd-Even Transposition Sort (MPI on multiple cores)
  
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
  
