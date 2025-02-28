#!/bin/bash

echo "Script starting"

# 2^10
sbatch quicksort_cuda.grace_job 1024 64 3
sbatch quicksort_cuda.grace_job 1024 128 3
sbatch quicksort_cuda.grace_job 1024 256 3
sbatch quicksort_cuda.grace_job 1024 512 3
sbatch quicksort_cuda.grace_job 1024 1024 3

# 2^12
sbatch quicksort_cuda.grace_job 4096 64 3
sbatch quicksort_cuda.grace_job 4096 128 3
sbatch quicksort_cuda.grace_job 4096 256 3
sbatch quicksort_cuda.grace_job 4096 512 3
sbatch quicksort_cuda.grace_job 4096 1024 3

# 2^14
sbatch quicksort_cuda.grace_job 16384 64 3
sbatch quicksort_cuda.grace_job 16384 128 3
sbatch quicksort_cuda.grace_job 16384 256 3
sbatch quicksort_cuda.grace_job 16384 512 3
sbatch quicksort_cuda.grace_job 16384 1024 3

# 2^16
sbatch quicksort_cuda.grace_job 65536 64 3
sbatch quicksort_cuda.grace_job 65536 128 3
sbatch quicksort_cuda.grace_job 65536 256 3
sbatch quicksort_cuda.grace_job 65536 512 3
sbatch quicksort_cuda.grace_job 65536 1024 3

# 2^18
sbatch quicksort_cuda.grace_job 262144 64 3
sbatch quicksort_cuda.grace_job 262144 128 3
sbatch quicksort_cuda.grace_job 262144 256 3
sbatch quicksort_cuda.grace_job 262144 512 3
sbatch quicksort_cuda.grace_job 262144 1024 3



echo "Script execution finished"