#!/bin/bash

numThreads=$1
inputType=$2

waitTime=2

for val in 65536 262144 1048576 4194304 16777216 67108864 268435456;
do
	sbatch mergesort.grace_job $numThreads $val $inputType
done
