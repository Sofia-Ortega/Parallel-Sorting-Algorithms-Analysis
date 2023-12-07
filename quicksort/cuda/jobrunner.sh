#!/bin/bash

echo "Script starting"

## 2^16
# sbatch quicksort_cuda.grace_job 65536 64 0
# sbatch quicksort_cuda.grace_job 65536 128 0
# sbatch quicksort_cuda.grace_job 65536 256 0
# sbatch quicksort_cuda.grace_job 65536 512 0
# sbatch quicksort_cuda.grace_job 65536 1024 0

## 2^18
# sbatch quicksort_cuda.grace_job 262144 64 0
# sbatch quicksort_cuda.grace_job 262144 128 0
sbatch quicksort_cuda.grace_job 262144 256 0
sbatch quicksort_cuda.grace_job 262144 512 0
sbatch quicksort_cuda.grace_job 262144 1024 0

## 2^20
# sbatch quicksort_cuda.grace_job 1048576 64 0
# sbatch quicksort_cuda.grace_job 1048576 128 0
sbatch quicksort_cuda.grace_job 1048576 256 0
sbatch quicksort_cuda.grace_job 1048576 512 0
sbatch quicksort_cuda.grace_job 1048576 1024 0

## 2^22
# sbatch quicksort_cuda.grace_job 4194304 64 0
sbatch quicksort_cuda.grace_job 4194304 128 0
sbatch quicksort_cuda.grace_job 4194304 256 0
sbatch quicksort_cuda.grace_job 4194304 512 0
sbatch quicksort_cuda.grace_job 4194304 1024 0

## 2^24
# sbatch quicksort_cuda.grace_job 16777216 64 0
sbatch quicksort_cuda.grace_job 16777216 128 0
sbatch quicksort_cuda.grace_job 16777216 256 0
sbatch quicksort_cuda.grace_job 16777216 512 0
sbatch quicksort_cuda.grace_job 16777216 1024 0

## 2^26
# sbatch quicksort_cuda.grace_job 67108864 64 0
sbatch quicksort_cuda.grace_job 67108864 128 0
sbatch quicksort_cuda.grace_job 67108864 256 0
sbatch quicksort_cuda.grace_job 67108864 512 0
sbatch quicksort_cuda.grace_job 67108864 1024 0

## 2^28
# sbatch quicksort_cuda.grace_job 268435456 64 0
sbatch quicksort_cuda.grace_job 268435456 128 0
sbatch quicksort_cuda.grace_job 268435456 256 0
sbatch quicksort_cuda.grace_job 268435456 512 0
sbatch quicksort_cuda.grace_job 268435456 1024 0

echo "Script execution finished"