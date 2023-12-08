#!/bin/bash

echo "Script starting"

# 2^16
# sbatch quicksort_cuda.grace_job 65536 64 2
# sbatch quicksort_cuda.grace_job 65536 128 2
# sbatch quicksort_cuda.grace_job 65536 256 2
# sbatch quicksort_cuda.grace_job 65536 512 2
# sbatch quicksort_cuda.grace_job 65536 1024 2

# 2^18
sbatch quicksort_cuda.grace_job 262144 64 2
# sbatch quicksort_cuda.grace_job 262144 128 2
# sbatch quicksort_cuda.grace_job 262144 256 2
# sbatch quicksort_cuda.grace_job 262144 512 2
# sbatch quicksort_cuda.grace_job 262144 1024 2

# 2^20
sbatch quicksort_cuda.grace_job 1048576 64 2
# sbatch quicksort_cuda.grace_job 1048576 128 2
# sbatch quicksort_cuda.grace_job 1048576 256 2
# sbatch quicksort_cuda.grace_job 1048576 512 2
# sbatch quicksort_cuda.grace_job 1048576 1024 2

# 2^22
sbatch quicksort_cuda.grace_job 4194304 64 2
# sbatch quicksort_cuda.grace_job 4194304 128 2
# sbatch quicksort_cuda.grace_job 4194304 256 2
# sbatch quicksort_cuda.grace_job 4194304 512 2
# sbatch quicksort_cuda.grace_job 4194304 1024 2

# 2^24
sbatch quicksort_cuda.grace_job 16777216 64 2
# sbatch quicksort_cuda.grace_job 16777216 128 2
# sbatch quicksort_cuda.grace_job 16777216 256 2
# sbatch quicksort_cuda.grace_job 16777216 512 2
# sbatch quicksort_cuda.grace_job 16777216 1024 2

# 2^26
sbatch quicksort_cuda.grace_job 67108864 64 2
# sbatch quicksort_cuda.grace_job 67108864 128 2
# sbatch quicksort_cuda.grace_job 67108864 256 2
# sbatch quicksort_cuda.grace_job 67108864 512 2
# sbatch quicksort_cuda.grace_job 67108864 1024 2

# 2^28
sbatch quicksort_cuda.grace_job 268435456 64 2
# sbatch quicksort_cuda.grace_job 268435456 128 2
# sbatch quicksort_cuda.grace_job 268435456 256 2
# sbatch quicksort_cuda.grace_job 268435456 512 2
# sbatch quicksort_cuda.grace_job 268435456 1024 2

echo "Script execution finished"