#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            #Do not propagate environment
#SBATCH --get-user-env=L         #Replicate login environment
#
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobName	 #Set the job name to "JobName"
#SBATCH --time=2:00:00           #Set the wall clock limit
#SBATCH --nodes=16               #Request nodes
#SBATCH --ntasks-per-node=4	 #Request 4 tasks/cores per node
#SBATCH --mem=8G                 #Request 8GB per node
#SBATCH --output=output.%j	 #Send stdout/err to "output.[jobID]"
#
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address
#
##First Executable Line
#
processes=$1
values=$2

module load intel/2020b       # load Intel software stack
module load CMake/3.12.1

CALI_CONFIG="spot(output=caliper-${values}.cali)" \
mpirun -np $processes ./oetsort $values

squeue -j $SLURM_JOBID
