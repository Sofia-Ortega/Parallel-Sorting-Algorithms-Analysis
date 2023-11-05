#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define ROOT 0

#pragma once

template<class T>
void printVals(T *vals, int count)
{
	//std::cout << std::fixed;
	//std::cout << std::setprecision(3);
	std::cout << "[";
	for (int i = 0; i < count - 1; ++i)
	{
		std::cout << vals[i] << ", ";
	}

	std::cout << vals[count-1] << "]" <<std::endl;
}

void fillValsRandParallel(double *arr, int valsPerProc, int seed);

void fillValsRandParallel(int *arr, int valsPerProc, int seed);

void fillValsRandSequential(double *arr, int count, int seed);

void fillValsRandSequential(int *arr, int count, int seed);

void fillValsSortedParallel(double *arr, int startVal, int valsPerProc);

void fillValsSortedParallel(int *arr, int startVal, int valsPerProc);

void fillValsSortedSequential(double *arr, int count);

void fillValsSortedSequential(int *arr, int count);

void fillValsReverseParallel(double *arr, int startVal, int valsPerProc);

void fillValsReverseParallel(int *arr, int startVal, int valsPerProc);

void fillValsReverseSequential(double *arr, int count);

void fillValsReverseSequential(int *arr, int count);

void genValues(int taskid, int numprocs, int numOfValues, bool isDoubles, void *out, int option = 0);
