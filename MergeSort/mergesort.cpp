#include "../InputGeneration/inputgen.h"

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
 * Link: geeksforgeeks.org/in-place-merge-sort
 * 
 */
void mergeInPlace(double *arr, int left, int mid, int right)
{
	int start = mid + 1;

	if (arr[mid] <= arr[start]) return;

	while (left <= mid && start <= right)
	{
		if (arr[left] <= arr[start]) ++left;
		else
		{
			int value = arr[start];
			int index = start;

			while (index != left)
			{
				arr[index] = arr[index - 1];
				--index;
			}

			arr[left] = value;

			++left;
			++mid;
			++start;
		}
	}
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

int main (int argc, char *argv[])
{
	//CALI_CXX_MARK_FUNCTION;

	//char * createSortedArray = "sorted_array_time";

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
	
	double genValuesTime = MPI_Wtime();
	genValues(taskid, numprocs, numberOfVals, 1, valuesDouble, option);
	genValuesTime = MPI_Wtime() - genValuesTime;

	if (taskid == ROOT)
	{
		// Time normal mergeSort
		double mergeSortTime = MPI_Wtime();
		mergeSort(valuesDouble, 0, numberOfVals - 1);
		mergeSortTime = MPI_Wtime() - mergeSortTime;

		// Time sorted check
		double sortedCheck = MPI_Wtime();
		bool sorted = isSorted(valuesDouble, numberOfVals);
		if (!sorted) printf("Array is not sorted\n");
		else printf("Array is sorted\n");
		sortedCheck = MPI_Wtime() - sortedCheck;

		//printArr(valuesDouble, numberOfVals);

		printf("genValues Time: %.3f\n", genValuesTime);
		printf("mergesort Time: %.3f\n", mergeSortTime);
		printf("sortcheck Time: %.3f\n", sortedCheck);
		delete[] valuesDouble;
	}

	/*
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

	*/

   	MPI_Finalize();
}
