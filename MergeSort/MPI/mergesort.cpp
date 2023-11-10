#include "../../InputGeneration/inputgen.h"
#include <cmath>
#include <string>

const char * parallel = "parallel";
const char * sequential = "sequential";
const char * genValuesTime = "data_init";
const char * correctness = "correctness_check";
const char * comp = "comp";
const char * compSmall = "comp_small";
const char * compLarge = "comp_large";
const char * commRegion = "comm";
const char * commSmall = "comm_small";
const char * commLarge = "comm_large";
const char * paraMergeTime = "para_merge_time";

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
	CALI_CXX_MARK_FUNCTION;

	int parent, rightChild, myHeight;
	double *half1, *half2, *mergeResult;

	myHeight = 0;
	CALI_MARK_BEGIN(comp);
	CALI_MARK_BEGIN(compLarge);
	CALI_MARK_BEGIN("Serial_Mergesort");
	mergeSort(localArray, 0, size - 1);
	CALI_MARK_END("Serial_Mergesort");
	CALI_MARK_END(compLarge);
	CALI_MARK_END(comp);
	half1 = localArray;

	while (myHeight < height)
	{
		parent = (id & (~(1 << myHeight)));

		if (parent == id) // left child
		{
			rightChild = (id | (1 << myHeight));

			// Get the second half from child
			half2 = new double[size];
			CALI_MARK_BEGIN(commRegion);
			CALI_MARK_BEGIN(commLarge);
			CALI_MARK_BEGIN("MPI_Recv");
			MPI_Recv(half2, size, MPI_DOUBLE, rightChild, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			CALI_MARK_END("MPI_Recv");
			CALI_MARK_END(commLarge);
			CALI_MARK_END(commRegion);

			// Merge both halves
			mergeResult = new double[2*size];
			CALI_MARK_BEGIN(comp);
			CALI_MARK_BEGIN(compLarge);
			CALI_MARK_BEGIN("Merging_Arrays");
			mergeParallel(half1, half2, mergeResult, size);
			CALI_MARK_END("Merging_Arrays");
			CALI_MARK_END(compLarge);
			CALI_MARK_END(comp);
			half1 = mergeResult;
			size *= 2;
			delete[] half2;
			mergeResult = nullptr;
			++myHeight;
		}
		else // right child
		{
			CALI_MARK_BEGIN(commRegion);
			CALI_MARK_BEGIN(commLarge);
			CALI_MARK_BEGIN("MPI_Send");
			MPI_Send(half1, size, MPI_DOUBLE, parent, 0, MPI_COMM_WORLD);
			CALI_MARK_END("MPI_Send");
			CALI_MARK_END(commLarge);
			CALI_MARK_END(commRegion);
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
	
	CALI_MARK_BEGIN(sequential);
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
		//CALI_MARK_BEGIN(seqMergesTime);
		CALI_MARK_BEGIN(comp);
		CALI_MARK_BEGIN(compLarge);
		mergeSort(valuesDouble, 0, numberOfVals - 1);
		CALI_MARK_END(compLarge);
		CALI_MARK_END(comp);
		//CALI_MARK_END(seqMergesTime);
		mergeSortTime = MPI_Wtime() - mergeSortTime;

		// Time sorted check
		double sortedCheck = MPI_Wtime();
		// TODO time validation
		//CALI_MARK_BEGIN(valValuesTime);
		CALI_MARK_BEGIN(correctness);
		bool sorted = isSorted(valuesDouble, numberOfVals);
		CALI_MARK_END(correctness);
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

	CALI_MARK_END(sequential);
	/* ********** Parallel Mergesort ********** */
	/*
	 * A lot of the code below was found from the same
	 * link used for the MPI mergesort.
	 *
	 * url: Link: selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
	 *
	 */

	CALI_MARK_BEGIN(paraMergeTime);

	genValuesTimes = MPI_Wtime();
	CALI_MARK_BEGIN(genValuesTime);
	genValues(taskid, numprocs, numberOfVals, 1, valuesDouble, option);
	CALI_MARK_END(genValuesTime);
	genValuesTimes = MPI_Wtime() - genValuesTimes;

	int height = log2(numprocs);

	int localArraySize = numberOfVals / numprocs;
	double *localArray = new double[localArraySize];
	CALI_MARK_BEGIN(commRegion);
	CALI_MARK_BEGIN(commLarge);
	CALI_MARK_BEGIN("MPI_Scatter");
	MPI_Scatter(valuesDouble, localArraySize, MPI_DOUBLE, localArray, localArraySize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	CALI_MARK_END("MPI_Scatter");
	CALI_MARK_END(commLarge);
	CALI_MARK_END(commRegion);

	double mergeParallelTime = MPI_Wtime();
	double rootTime;
	double procTime;
	if (taskid == ROOT)
	{
		//printArr(valuesDouble, numberOfVals);
		rootTime = MPI_Wtime();
		//CALI_MARK_BEGIN(comp);
		//CALI_MARK_BEGIN(compLarge);
		valuesDouble = mergeSortParallel(height, taskid, localArray, localArraySize, MPI_COMM_WORLD, valuesDouble);
		//CALI_MARK_END(compLarge);
		//CALI_MARK_END(comp);
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
	CALI_MARK_BEGIN(commRegion);
	CALI_MARK_BEGIN(commLarge);
	CALI_MARK_BEGIN("MPI_Reduce");
	MPI_Reduce(&mergeParallelTime, &totalMergeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	CALI_MARK_END("MPI_Reduce");
	CALI_MARK_END(commLarge);
	CALI_MARK_END(commRegion);

	if (taskid == ROOT)
	{
		// Time sorted check
		double sortedCheck = MPI_Wtime();
		CALI_MARK_BEGIN(correctness);
		bool sorted = isSorted(valuesDouble, numberOfVals);
		CALI_MARK_END(correctness);
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

	CALI_MARK_END(paraMergeTime);
	// Create caliper ConfigManager object
	cali::ConfigManager mgr;
	mgr.start();

	adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Mergesort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 8); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", numberOfVals); // The number of elements in input dataset (1000)
	std::string inputType = "Random";
	if (option == 1) inputType = "Sorted";
	else if (option == 2) inputType = "ReverseSorted";
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numprocs); // The number of processors (MPI ranks)
    adiak::value("group_num", 23); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online/Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

   	// Flush Caliper output before finalizing MPI
   	mgr.stop();
   	mgr.flush();

   	MPI_Finalize();
}
