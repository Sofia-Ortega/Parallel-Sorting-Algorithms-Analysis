#!/bin/bash

echo "Script starting"

## 2^18
sbatch quicksort_cuda.grace_job 262144 128 0
sbatch quicksort_cuda.grace_job 262144 128 1
sbatch quicksort_cuda.grace_job 262144 128 2
sbatch quicksort_cuda.grace_job 262144 128 3

echo "Script execution finished"