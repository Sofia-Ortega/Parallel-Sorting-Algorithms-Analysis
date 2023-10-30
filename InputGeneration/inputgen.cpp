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
		printf("%.3f, ", vals[i]);
		//std::cout << vals[i] << ", ";
	}
	std::cout << vals[count-1] << "]" << std::endl;
}

void fillValsRandParallel(double *arr, int valsPerProc, int seed)
{
    srand((unsigned)seed);
	for (int i = 0; i < valsPerProc; ++i)
	{
		double randVal = (double)(rand() % 1000000);
		randVal += (randVal / 1000000);
		arr[i] = randVal;
	}
}

void fillValsRandSequential(double *arr, int count, int seed)
{
    srand((unsigned) seed);
	for (int i = 0; i < count; ++i)
	{
		double randVal = (double)(rand() % 1000000);
		randVal += (randVal / 1000000);
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

void genValues(int taskid, int numprocs, int numOfValues, bool isDoubles, void *out, int option = 0)
{
	int valsPerProc = numOfValues / numprocs;
	
	if (isDoubles)
	{
		double *inputPerProcRand; 
		double *inputPerProcSorted; 
		double *inputPerProcReverse;

		if (option == 0)
		{
			inputPerProcRand = new double[valsPerProc];
			fillValsRandParallel(inputPerProcRand, valsPerProc, 10 + taskid);
			MPI_Gather(inputPerProcRand, valsPerProc, MPI_DOUBLE, (double*)out, valsPerProc, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		}
		else if (option == 1)
		{
			int startVal = (taskid * valsPerProc);
			inputPerProcSorted = new double[valsPerProc];
			fillValsSortedParallel(inputPerProcSorted, startVal, valsPerProc);
			MPI_Gather(inputPerProcSorted, valsPerProc, MPI_DOUBLE, (double*)out, valsPerProc, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		}
		else
		{
			int startVal = ((numprocs-1) * valsPerProc) - (taskid * valsPerProc);
			inputPerProcReverse = new double[valsPerProc];
			fillValsReverseParallel(inputPerProcReverse, startVal, valsPerProc);
			MPI_Gather(inputPerProcReverse, valsPerProc, MPI_DOUBLE, (double*)out, valsPerProc, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		}
	}
	else
	{
	
	}
}

int main (int argc, char *argv[])
{
	CALI_CXX_MARK_FUNCTION;

	char * createSortedArray = "sorted_array_time";

	int numberOfVals = atoi(argv[1]);
	bool isDoubles = (atoi(argv[2]) == 1);
	int option = atoi(argv[3]);

	int taskid;
	int numprocs;
    
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	// Generate sorted array of doubles
	/*
	int valsPerProc = numberOfVals / numprocs;
	double *inputPerProcRand = new double[valsPerProc];
	double *inputPerProcSorted = new double[valsPerProc];
	double *inputPerProcReverse = new double[valsPerProc];
	*/

	//int startVal = ((numprocs-1) * valsPerProc) - (taskid * valsPerProc);
	void *values;
	
	//double *inputValsRand = nullptr;
	//double *inputValsSorted = nullptr;
	//double *inputValsReverse = nullptr;
	double test;
	if (taskid == ROOT)
	{
		if (isDoubles) values = new double[numberOfVals];	
		else values = new int[numberOfVals];
		test = MPI_Wtime();
	}
	
	double genValuesTime = MPI_Wtime();
	genValues(taskid, numprocs, numberOfVals, isDoubles, values, option);
	genValuesTime = MPI_Wtime() - genValuesTime;

	if (taskid == ROOT)
	{
		test = MPI_Wtime() - test;
		double *valsSeqRand = new double[numberOfVals];
		double randArrayTimeSeq = MPI_Wtime();
		fillValsRandSequential(valsSeqRand, numberOfVals, numprocs*3);
		randArrayTimeSeq = MPI_Wtime() - randArrayTimeSeq;

		printf("genValues Parallel Time: %.3f\n", genValuesTime);
		printf("genValues Sequence Time: %.3f\n", randArrayTimeSeq);
		printf("test Time: %.3f\n", test);

		//printVals<double>((double*)values, numberOfVals);
		delete[] valsSeqRand;
	}

	/*
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
	*/
	/*
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
	*/

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
