#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            #Do not propagate environment
#SBATCH --get-user-env=L         #Replicate login environment
#
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobName       #Set the job name to "JobName"
#SBATCH --time=0:03:00           #Set the wall clock limit
#SBATCH --nodes=40               #Request nodes
#SBATCH --ntasks-per-node=32     #Request 4 tasks/cores per node
#SBATCH --mem=16G                 #Request 8GB per node 
#SBATCH --output=output.%j       #Send stdout/err to "output.[jobID]" 
#
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address 
#
##First Executable Line
#
array_size=$1
processes=$2
option=$3

module load intel/2020b       # load Intel software stack
module load CMake/3.12.1

CALI_CONFIG="spot(output=p${processes}-a${array_size}-o${option}.cali, \
  time.variance)" \
mpirun -np $processes ./quicksort_mpi $array_size $option