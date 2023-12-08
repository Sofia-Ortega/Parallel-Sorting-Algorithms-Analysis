#!/bin/bash

echo "Script starting"

## input size thread-count option - random
# 2^16
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 65536 2 2
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 65536 4 2
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 65536 8 2
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 65536 16 2
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 65536 32 2
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 65536 64 2
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 65536 128 2
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 65536 256 2
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 65536 512 2
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 65536 1024 2

# 2^18
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 262144 2 2
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 262144 4 2
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 262144 8 2
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 262144 16 2
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 262144 32 2
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 262144 64 2
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 262144 128 2
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 262144 256 2
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 262144 512 2
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 262144 1024 2

# 2^20
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 1048576 2 2
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 1048576 4 2
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 1048576 8 2
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 1048576 16 2
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 1048576 32 2
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 1048576 64 2
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 1048576 128 2
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 1048576 256 2
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 1048576 512 2
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 1048576 1024 2

# 2^22
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 4194304 2 2
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 4194304 4 2
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 4194304 8 2
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 4194304 16 2
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 4194304 32 2
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 4194304 64 2
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 4194304 128 2
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 4194304 256 2
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 4194304 512 2
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 4194304 1024 2

# 2^24
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 16777216 2 2
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 16777216 4 2
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 16777216 8 2
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 16777216 16 2
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 16777216 32 2
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 16777216 64 2
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 16777216 128 2
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 16777216 256 2
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 16777216 512 2
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 16777216 1024 2

# 2^26
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 67108864 2 2
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 67108864 4 2
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 67108864 8 2
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 67108864 16 2
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 67108864 32 2
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 67108864 64 2
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 67108864 128 2
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 67108864 256 2
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 67108864 512 2
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 67108864 1024 2

# 2^28
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 268435456 2 2
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 268435456 4 2
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 268435456 8 2
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 268435456 16 2
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 268435456 32 2
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 268435456 64 2
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 268435456 128 2
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 268435456 256 2
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 268435456 512 2
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 268435456 1024 2


echo "Script execution finished"
