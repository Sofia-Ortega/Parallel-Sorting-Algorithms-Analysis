/******************************************************************************
* FILE: inputgen.cpp
* DESCRIPTION:  
*   Generates large arrays of numbers in parallel to be used by sorting algos.
* AUTHOR: Will Thompson
******************************************************************************/
#include "inputgen.h"

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

void fillValsRandParallel(int *arr, int valsPerProc, int seed)
{
    srand((unsigned)seed);
	for (int i = 0; i < valsPerProc; ++i)
	{
		arr[i] = (rand() % 1000000);
	}
}

void fillValsRandSequential(double *arr, int count, int seed)
{
    srand((unsigned) seed);
	for (int i = 0; i < count; ++i)
	{
		double randVal = (double)(rand() % 1000000);
		randVal += (randVal / 1000000);
		arr[i] = randVal;
	}
}

void fillValsRandSequential(int *arr, int count, int seed)
{
    srand((unsigned) seed);
	for (int i = 0; i < count; ++i)
	{
		arr[i] = (rand() % 1000000);
	}
}

void fillValsSortedParallel(double *arr, int startVal, int valsPerProc)
{
	for (int i = 0; i < valsPerProc; ++i)
	{
		arr[i] = (double) (i + startVal);
	}
}

void fillValsSortedParallel(int *arr, int startVal, int valsPerProc)
{
	for (int i = 0; i < valsPerProc; ++i)
	{
		arr[i] = (i + startVal);
	}
}

void fillValsSortedSequential(double *arr, int count)
{
	for (int i = 0; i < count; ++i)
	{
		arr[i] = (double) (i);
	}
}

void fillValsSortedSequential(int *arr, int count)
{
	for (int i = 0; i < count; ++i)
	{
		arr[i] = (i);
	}
}

void fillValsReverseParallel(double *arr, int startVal, int valsPerProc)
{
	for (int i = valsPerProc-1; i >= 0; --i)
	{
		arr[i] = (double) (((valsPerProc-1) - i) + startVal);
	}
}

void fillValsReverseParallel(int *arr, int startVal, int valsPerProc)
{
	for (int i = valsPerProc-1; i >= 0; --i)
	{
		arr[i] = (((valsPerProc-1) - i) + startVal);
	}
}

void fillValsReverseSequential(double *arr, int count)
{
	for (int i = count-1; i >= 0; --i)
	{
		arr[i] = (double) (i);
	}
}

void fillValsReverseSequential(int *arr, int count)
{
	for (int i = count-1; i >= 0; --i)
	{
		arr[i] = (i);
	}
}

void genValues(int taskid, int numprocs, int numOfValues, bool isDoubles, void *out, int option)
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
		int *inputPerProcRand; 
		int *inputPerProcSorted; 
		int *inputPerProcReverse;

		if (option == 0)
		{
			inputPerProcRand = new int[valsPerProc];
			fillValsRandParallel(inputPerProcRand, valsPerProc, 10 + taskid);
			MPI_Gather(inputPerProcRand, valsPerProc, MPI_INT, (int*)out, valsPerProc, MPI_INT, ROOT, MPI_COMM_WORLD);
		}
		else if (option == 1)
		{
			int startVal = (taskid * valsPerProc);
			inputPerProcSorted = new int[valsPerProc];
			fillValsSortedParallel(inputPerProcSorted, startVal, valsPerProc);
			MPI_Gather(inputPerProcSorted, valsPerProc, MPI_INT, (int*)out, valsPerProc, MPI_INT, ROOT, MPI_COMM_WORLD);
		}
		else
		{
			int startVal = ((numprocs-1) * valsPerProc) - (taskid * valsPerProc);
			inputPerProcReverse = new int[valsPerProc];
			fillValsReverseParallel(inputPerProcReverse, startVal, valsPerProc);
			MPI_Gather(inputPerProcReverse, valsPerProc, MPI_INT, (int*)out, valsPerProc, MPI_INT, ROOT, MPI_COMM_WORLD);
		}
	}
}
