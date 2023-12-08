#!/bin/bash

echo "Script starting"

## input size thread-count option - random
# 2^16
# sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 65536 2 3
# sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 65536 4 3
# sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 65536 8 3
# sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 65536 16 3
# sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 65536 32 3
# sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 65536 64 3
# sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 65536 128 3
# sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 65536 256 3
# sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 65536 512 3
# sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 65536 1024 3

# 2^18
# sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 262144 2 3
# sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 262144 4 3
# sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 262144 8 3
# sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 262144 16 3
# sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 262144 32 3
# sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 262144 64 3
# sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 262144 128 3
# sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 262144 256 3
# sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 262144 512 3
# sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 262144 1024 3

# 2^20
# sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 1048576 2 3
# sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 1048576 4 3
# sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 1048576 8 3
# sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 1048576 16 3
# sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 1048576 32 3
# sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 1048576 64 3
# sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 1048576 128 3
# sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 1048576 256 3
# sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 1048576 512 3
# sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 1048576 1024 3

# # 2^22
# sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 4194304 2 3
# sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 4194304 4 3
# sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 4194304 8 3
# sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 4194304 16 3
# sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 4194304 32 3
# sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 4194304 64 3
# sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 4194304 128 3
# sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 4194304 256 3
# sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 4194304 512 3
# sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 4194304 1024 3

# # 2^24
# sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 16777216 2 3
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 16777216 4 3
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 16777216 8 3
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 16777216 16 3
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 16777216 32 3
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 16777216 64 3
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 16777216 128 3
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 16777216 256 3
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 16777216 512 3
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 16777216 1024 3

# 2^26
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 67108864 2 3
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 67108864 4 3
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 67108864 8 3
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 67108864 16 3
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 67108864 32 3
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 67108864 64 3
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 67108864 128 3
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 67108864 256 3
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 67108864 512 3
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 67108864 1024 3

# 2^28
sbatch --nodes 1 --ntasks-per-node 2 mpi.grace_job 268435456 2 3
sbatch --nodes 1 --ntasks-per-node 4 mpi.grace_job 268435456 4 3
sbatch --nodes 1 --ntasks-per-node 8 mpi.grace_job 268435456 8 3
sbatch --nodes 1 --ntasks-per-node 16 mpi.grace_job 268435456 16 3
sbatch --nodes 1 --ntasks-per-node 32 mpi.grace_job 268435456 32 3
sbatch --nodes 2 --ntasks-per-node 32 mpi.grace_job 268435456 64 3
sbatch --nodes 4 --ntasks-per-node 32 mpi.grace_job 268435456 128 3
sbatch --nodes 8 --ntasks-per-node 32 mpi.grace_job 268435456 256 3
sbatch --nodes 16 --ntasks-per-node 32 mpi.grace_job 268435456 512 3
sbatch --nodes 32 --ntasks-per-node 32 mpi.grace_job 268435456 1024 3


echo "Script execution finished"
