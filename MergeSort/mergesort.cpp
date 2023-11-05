#include "../InputGeneration/inputgen.h"
#include <cmath>

bool isSorted(double *arr, int count)
{
	for (int i = 0; i < count - 1; ++i)
	{
		if (arr[i] > arr[i + 1]) return false;	
	}

	return true;
}

/*
 * This mergeSort function was found on geeksforgeeks.org
 *
 * Link: geeksforgeeks.org/merge-sort
 * 
 */
void merge(double *arr, int left, int mid, int right)
{
	int const subArrayOne = mid - left + 1;
	int const subArrayTwo = right - mid;

	auto *leftArray = new double[subArrayOne], *rightArray = new double[subArrayTwo];

	for (int i = 0; i < subArrayOne; ++i) leftArray[i] = arr[left + i];
	for (int i = 0; i < subArrayTwo; ++i) rightArray[i] = arr[mid + 1 + i];

	auto indexOfSubArrayOne = 0, indexOfSubArrayTwo = 0;
	int indexOfMergedArray = left;

	while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo)
	{
		if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo])
		{
			arr[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
			++indexOfSubArrayOne;
		}
		else
		{
			arr[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
			++indexOfSubArrayTwo;
		}

		++indexOfMergedArray;
	}

	while (indexOfSubArrayOne < subArrayOne)
	{
		arr[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
		++indexOfSubArrayOne;
		++indexOfMergedArray;
	}

	while (indexOfSubArrayTwo < subArrayTwo)
	{
		arr[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
		++indexOfSubArrayTwo;
		++indexOfMergedArray;;
	}

	delete[] leftArray;
	delete[] rightArray;
}

/*
 * This mergeSort function was found on geeksforgeeks.org
 *
 * Link: geeksforgeeks.org/merge-sort
 * 
 */
void mergeSort(double *arr, int begin, int end)
{
	if (begin >= end) return;
	
	int mid = begin + (end - begin) / 2;
	mergeSort(arr, begin, mid);
	mergeSort(arr, mid + 1, end);
	merge(arr, begin, mid, end);
}

void printArr(double *arr, int count)
{
	std::cout << "[";
	for (int i = 0; i < count - 1; ++i) printf("%.3f, ", arr[i]);
	std::cout << arr[count-1] << "]" << std::endl;
}

/*
 * This merging function was found on geeksforgeeks.org
 *
 * Link: geeksforgeeks.org/merge-sort
 * 
 */
void mergeParallel(double *arr1, double *arr2, double *arr3, int size)
{
	int i = 0, j = 0, k = 0;

	while (i < size && j < size)
	{
		if (arr1[i] < arr2[j]) arr3[k++] = arr1[i++];
		else arr3[k++] = arr2[j++];
	}

	while (i < size) arr3[k++] = arr1[i++];
	while (j < size) arr3[k++] = arr2[j++];

}

/*
 * This mergeSort function was found on selkie-macalester.org
 *
 * Link: selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
 * 
 */
double* mergeSortParallel(int height, int id, double *localArray, int size, MPI_Comm comm, double *globalArray)
{
	int parent, rightChild, myHeight;
	double *half1, *half2, *mergeResult;

	myHeight = 0;
	mergeSort(localArray, 0, size - 1);
	half1 = localArray;

	while (myHeight < height)
	{
		parent = (id & (~(1 << myHeight)));

		if (parent == id) // left child
		{
			rightChild = (id | (1 << myHeight));

			// Get the second half from child
			half2 = new double[size];
			MPI_Recv(half2, size, MPI_DOUBLE, rightChild, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Merge both halves
			mergeResult = new double[2*size];
			mergeParallel(half1, half2, mergeResult, size);
			half1 = mergeResult;
			size *= 2;
			delete[] half2;
			mergeResult = nullptr;
			++myHeight;
		}
		else // right child
		{
			MPI_Send(half1, size, MPI_DOUBLE, parent, 0, MPI_COMM_WORLD);
			if (myHeight != 0) delete[] half1;
			myHeight = height;
		}
	}

	if (id == ROOT)
	{
		globalArray = half1;
	}

	return globalArray;
}

int main (int argc, char *argv[])
{
	CALI_CXX_MARK_FUNCTION;

	const char * genValuesTime = "gen_values_time";
	const char * seqMergesTime = "seq_merges_time";
	const char * valValuesTime = "val_values_time";
	const char * paraMergeTime = "para_merge_time";

	int numberOfVals = atoi(argv[1]);
	int option = atoi(argv[2]);

	int taskid;
	int numprocs;
    
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	double *valuesDouble;
	
	if (taskid == ROOT)
	{
		valuesDouble = new double[numberOfVals];
	}

	/* ********** Sequential Mergesort ********** */
	
	double genValuesTimes = MPI_Wtime();
	// TODO time the input gen
	CALI_MARK_BEGIN(genValuesTime);
	genValues(taskid, numprocs, numberOfVals, 1, valuesDouble, option);
	CALI_MARK_END(genValuesTime);
	genValuesTimes = MPI_Wtime() - genValuesTimes;

	if (taskid == ROOT)
	{
		// Time normal mergeSort
		double mergeSortTime = MPI_Wtime();
		// TODO time sequence mergesort
		CALI_MARK_BEGIN(seqMergesTime);
		mergeSort(valuesDouble, 0, numberOfVals - 1);
		CALI_MARK_END(seqMergesTime);
		mergeSortTime = MPI_Wtime() - mergeSortTime;

		// Time sorted check
		double sortedCheck = MPI_Wtime();
		// TODO time validation
		CALI_MARK_BEGIN(valValuesTime);
		bool sorted = isSorted(valuesDouble, numberOfVals);
		CALI_MARK_END(valValuesTime);
		if (!sorted) printf("Array is not sorted\n");
		else printf("Array is sorted\n");
		sortedCheck = MPI_Wtime() - sortedCheck;

		//printArr(valuesDouble, numberOfVals);

		printf("Sequential Mergesort:\n");
		printf("\tgenValues Time: %.3f\n", genValuesTimes);
		printf("\tmergesort Time: %.3f\n", mergeSortTime);
		printf("\tsortcheck Time: %.3f\n", sortedCheck);
		//delete[] valuesDouble;
		//printArr(valuesDouble, numberOfVals);
	}

	/* ********** Parallel Mergesort ********** */
	/*
	 * A lot of the code below was found from the same
	 * link used for the MPI mergesort.
	 *
	 * url: Link: selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
	 *
	 */

	genValuesTimes = MPI_Wtime();
	// TODO time the input gen
	genValues(taskid, numprocs, numberOfVals, 1, valuesDouble, option);
	genValuesTimes = MPI_Wtime() - genValuesTimes;

	int height = log2(numprocs);

	int localArraySize = numberOfVals / numprocs;
	double *localArray = new double[localArraySize];
	MPI_Scatter(valuesDouble, localArraySize, MPI_DOUBLE, localArray, localArraySize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double mergeParallelTime = MPI_Wtime();
	double rootTime;
	double procTime;
	if (taskid == ROOT)
	{
		//printArr(valuesDouble, numberOfVals);
		rootTime = MPI_Wtime();
		valuesDouble = mergeSortParallel(height, taskid, localArray, localArraySize, MPI_COMM_WORLD, valuesDouble);
		rootTime = MPI_Wtime() - rootTime;
	}
	else
	{
		procTime = MPI_Wtime();
		mergeSortParallel(height, taskid, localArray, localArraySize, MPI_COMM_WORLD, NULL);
		procTime = MPI_Wtime() - procTime;
	}

	mergeParallelTime = MPI_Wtime() - mergeParallelTime;
	double totalMergeTime;
	MPI_Reduce(&mergeParallelTime, &totalMergeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (taskid == ROOT)
	{
		// Time sorted check
		double sortedCheck = MPI_Wtime();
		bool sorted = isSorted(valuesDouble, numberOfVals);
		if (!sorted) printf("Array is not sorted\n");
		else printf("Array is sorted\n");
		sortedCheck = MPI_Wtime() - sortedCheck;

		printf("Parallel Mergesort:\n");
		printf("\tgenValues Time: %.3f\n", genValuesTimes);
		printf("\tRoot Time: %.3f\n", rootTime);
		printf("\tProc Time: %.3f\n", procTime);
		printf("\tMergesort Time: %.3f\n", totalMergeTime);
		printf("\tSortcheck Time: %.3f\n", sortedCheck);
		
		delete[] valuesDouble;
	}

	delete[] localArray;


	// Create caliper ConfigManager object
	cali::ConfigManager mgr;
	mgr.start();

   	adiak::init(NULL);
   	adiak::user();
   	adiak::clustername();
   	adiak::value("num_procs", numprocs);
   	adiak::value("num_values", numberOfVals);
   	adiak::value("program_name", "MergeSort");

   	// Flush Caliper output before finalizing MPI
   	mgr.stop();
   	mgr.flush();

   	MPI_Finalize();
}
