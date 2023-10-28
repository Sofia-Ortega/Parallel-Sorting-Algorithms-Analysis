/******************************************************************************
* FILE: inputgen.cpp
* DESCRIPTION:  
*   Generates large arrays of numbers in parallel to be used by sorting algos.
* AUTHOR: Will Thompson
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define ROOT 0

template<class T>
void printVals(T *vals, int count)
{
	std::cout << "[";
	for (int i = 0; i < count - 1; ++i)
	{
		std::cout << vals[i] << ", ";
	}
	std::cout << vals[count-1] << "]" << std::endl;
}

void fillValsRandParallel(double *arr, int valsPerProc)
{
    srand((unsigned) time(NULL));
	for (int i = 0; i < valsPerProc; ++i)
	{
		arr[i] = (double) (rand() % 1000000);
	}
}

void fillValsRandSequential(double *arr, int count)
{
    srand((unsigned) time(NULL));
	for (int i = 0; i < count; ++i)
	{
		arr[i] = (double) (rand() % 1000000);
	}
}

void fillValsSortedParallel(double *arr, int startVal, int valsPerProc)
{
	for (int i = 0; i < valsPerProc; ++i)
	{
		arr[i] = (double) (i + startVal);
	}
}

void fillValsSortedSequential(double *arr, int count)
{
	for (int i = 0; i < count; ++i)
	{
		arr[i] = (double) (i);
	}
}

void fillValsReverseParallel(double *arr, int startVal, int valsPerProc)
{
	for (int i = valsPerProc-1; i >= 0; --i)
	{
		arr[i] = (double) (((valsPerProc-1) - i) + startVal);
	}
}

void fillValsReverseSequential(double *arr, int count)
{
	for (int i = count-1; i >= 0; --i)
	{
		arr[i] = (double) (i);
	}
}

int main (int argc, char *argv[])
{
	CALI_CXX_MARK_FUNCTION;

	char * createSortedArray = "sorted_array_time";

	int numberOfVals = atoi(argv[1]);

	int taskid;
	int numprocs;
    
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	// Generate sorted array of doubles
	int valsPerProc = numberOfVals / numprocs;
	double *inputPerProcRand = new double[valsPerProc];
	double *inputPerProcSorted = new double[valsPerProc];
	double *inputPerProcReverse = new double[valsPerProc];

	int startVal = ((numprocs-1) * valsPerProc) - (taskid * valsPerProc);
	
	double *inputValsRand = nullptr;
	double *inputValsSorted = nullptr;
	double *inputValsReverse = nullptr;
	if (taskid == ROOT)
	{
		inputValsRand = new double[numberOfVals];	
		inputValsSorted = new double[numberOfVals];	
		inputValsReverse = new double[numberOfVals];	
	}

	// Create sorted values array in parallel

	double sortedArrayTime = MPI_Wtime();

	fillValsSortedParallel(inputPerProcSorted, startVal, valsPerProc);

	MPI_Gather(inputPerProcSorted, valsPerProc, MPI_DOUBLE, inputValsSorted, valsPerProc, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	sortedArrayTime = MPI_Wtime() - sortedArrayTime;

	// Create random values array in parallel

	CALI_MARK_BEGIN(createSortedArray);
    double randArrayTime = MPI_Wtime();

	fillValsRandParallel(inputPerProcRand, valsPerProc);

	MPI_Gather(inputPerProcRand, valsPerProc, MPI_DOUBLE, inputValsRand, valsPerProc, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	randArrayTime = MPI_Wtime() - randArrayTime;
	CALI_MARK_END(createSortedArray);

	// Create reverse sorted values array in parallel

	double reverseArrayTime = MPI_Wtime();

	fillValsReverseParallel(inputPerProcReverse, startVal, valsPerProc);

	MPI_Gather(inputPerProcReverse, valsPerProc, MPI_DOUBLE, inputValsReverse, valsPerProc, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	reverseArrayTime = MPI_Wtime() - reverseArrayTime;

	if (taskid == ROOT)
	{
		// Create sorted array sequentially
		double *valsSeqSorted = new double[numberOfVals];
		double sortedArrayTimeSeq = MPI_Wtime();
		fillValsSortedSequential(valsSeqSorted, numberOfVals);
		sortedArrayTimeSeq = MPI_Wtime() - sortedArrayTimeSeq;

		// Create random array sequentially
		double *valsSeqRand = new double[numberOfVals];
		double randArrayTimeSeq = MPI_Wtime();
		fillValsRandSequential(valsSeqRand, numberOfVals);
		randArrayTimeSeq = MPI_Wtime() - randArrayTimeSeq;

		// Create reverse sorted array sequentially
		double *valsSeqReverse = new double[numberOfVals];
		double reverseArrayTimeSeq = MPI_Wtime();
		fillValsReverseSequential(valsSeqReverse, numberOfVals);
		reverseArrayTimeSeq = MPI_Wtime() - reverseArrayTimeSeq;

		// Print times
		//printVals<double>(inputValsReverse, numberOfVals);
		printf("Sorted array times:\n\tParallel - %3.5f\n\tSequential - %3.5f\n", sortedArrayTime, sortedArrayTimeSeq);
		printf("Random array times:\n\tParallel - %3.5f\n\tSequential - %3.5f\n", randArrayTime, randArrayTimeSeq);
		printf("Reverse sorted  array times:\n\tParallel - %3.5f\n\tSequential - %3.5f\n", reverseArrayTime, reverseArrayTimeSeq);
		delete[] inputValsRand;
		delete[] inputValsSorted;
	}

	// Create caliper ConfigManager object
	cali::ConfigManager mgr;
	mgr.start();

   	adiak::init(NULL);
   	adiak::user();
   	adiak::launchdate();
   	adiak::libraries();
   	adiak::cmdline();
   	adiak::clustername();
   	adiak::value("num_procs", numprocs);
   	adiak::value("program_name", "Input Generation");

   	// Flush Caliper output before finalizing MPI
   	mgr.stop();
   	mgr.flush();

   	MPI_Finalize();
}
