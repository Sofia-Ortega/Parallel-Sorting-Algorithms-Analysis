#!/bin/bash

inputSize=$1
inputType=$2

waitTime=10

if [ $inputSize == 0 ];
then
	for val in 65536 262144 1048576 4194304 16777216 67108864 268435456;
	do
		for numprocs in 2 4 8 16 32 64 128 256 512;
		do
			sbatch mergesort.grace_job $numprocs $val $inputType

			sleep $waitTime
		done
	done
else
	for numprocs in 2 4 8 16 32 64 128 256 512;
	do
		sbatch mergesort.grace_job $numprocs $inputSize $inputType

		sleep $waitTime
	done
fi


