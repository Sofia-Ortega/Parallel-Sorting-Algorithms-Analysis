/**
 * -------------------- SOURCE -----------------------------------
 * Code: https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/
 * Author: Ashutosh Soni
 * University: MNIT Jaipur
 * Date: December 20, 2012
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "inputgen.h"

using namespace std;

#define SEED 10

const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

// Function to swap two numbers
void swap(int *arr, int i, int j)
{
    // cout << "Swapping " << arr[i] << " and " << arr[j] << endl;
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

bool isSorted(int *arr, int count)
{
    for (int i = 0; i < count - 1; ++i)
    {
        if (arr[i] > arr[i + 1])
            return false;
    }

    return true;
}

void quicksort(int *arr, int start, int end)
{
    int pivot, index;

    // Base Case
    if (end <= 1)
        return;

    // Pick pivot and swap with first
    // element Pivot is middle element
    pivot = arr[start + end / 2];
    swap(arr, start, start + end / 2);

    // Partitioning Steps
    index = start;

    // Iterate over the range [start, end]
    for (int i = start + 1; i < start + end; i++)
    {

        // Swap if the element is less
        // than the pivot element
        if (arr[i] < pivot)
        {
            index++;
            swap(arr, i, index);
        }
    }

    // Swap the pivot into place
    swap(arr, start, index);

    // Recursive Call for sorting
    // of quick sort function
    quicksort(arr, start, index - start);
    quicksort(arr, index + 1, start + end - index - 1);
}

// Function that merges the two arrays
int *merge(int *arr1, int n1, int *arr2, int n2)
{
    int *result = (int *)malloc((n1 + n2) * sizeof(int));
    int i = 0;
    int j = 0;
    int k;

    for (k = 0; k < n1 + n2; k++)
    {
        if (i >= n1)
        {
            result[k] = arr2[j];
            j++;
        }
        else if (j >= n2)
        {
            result[k] = arr1[i];
            i++;
        }

        // Indices in bounds as i < n1
        // && j < n2
        else if (arr1[i] < arr2[j])
        {
            result[k] = arr1[i];
            i++;
        }

        // v2[j] <= v1[i]
        else
        {
            result[k] = arr2[j];
            j++;
        }
    }
    return result;
}

// Driver Code
int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;


    int number_of_elements = atoi(argv[1]);

    int printArray = 1;
    if(argc == 3) 
        printArray = atoi(argv[2]);

    int *data = NULL;
    int chunk_size, own_chunk_size;
    int *chunk;
    MPI_Status status;

    double dataInitTime, barrierTime, commSmallTime, commLargeTime, compSmallTime, compLargeTime, correctTime, totalTime;

    int number_of_process, rank_of_process;
    // Initialize the MPI environment
    int rc = MPI_Init(&argc, &argv);
    totalTime = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);

    if(rank_of_process == 0) {
        printf("Sorting %i elements with %i processes\n", number_of_elements, number_of_process);
    }

    CALI_MARK_BEGIN(data_init);
    dataInitTime = MPI_Wtime();
    if (rank_of_process == 0)
    {
        // create a array of size number_of_elements of random integers
        data = (int *)malloc(number_of_elements * sizeof(int));
        fillValsRandParallel(data, number_of_elements, SEED);
    }
    dataInitTime = MPI_Wtime() - dataInitTime;
    CALI_MARK_END(data_init);

    // Blocks all process until reach this point
    barrierTime = MPI_Wtime();

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN("MPI_Barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Barrier");
    CALI_MARK_END(comm);

    barrierTime = MPI_Wtime() - barrierTime;

    // BroadCast the Size to all the
    // process from root process
    commSmallTime = MPI_Wtime();

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN("MPI_Bcast");
    MPI_Bcast(&number_of_elements, 1, MPI_INT, 0,
              MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Bcast");
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    commSmallTime = MPI_Wtime() - commSmallTime;


    // Computing chunk size
    chunk_size = (number_of_elements % number_of_process == 0)
                     ? (number_of_elements / number_of_process)
                     : number_of_elements / (number_of_process - 1);

    // Calculating total size of chunk
    // according to bits
    chunk = (int *)malloc(chunk_size * sizeof(int));

    // Scatter the chuck size data to all process
    commLargeTime = MPI_Wtime();

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(data, chunk_size, MPI_INT, chunk,
                chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    commLargeTime = MPI_Wtime() - commLargeTime;


    free(data);
    data = NULL;


    // Compute size of own chunk and
    // then sort them
    // using quick sort

    own_chunk_size = (number_of_elements >= chunk_size * (rank_of_process + 1))
                         ? chunk_size
                         : (number_of_elements - chunk_size * rank_of_process);

    // Sorting array with quick sort for every
    // chunk as called by process
    compLargeTime = MPI_Wtime();

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    quicksort(chunk, 0, own_chunk_size);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    compLargeTime = MPI_Wtime() - compLargeTime;


    for (int step = 1; step < number_of_process;
         step = 2 * step)
    {
        if (rank_of_process % (2 * step) != 0)
        {
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_large);
            CALI_MARK_BEGIN("MPI_Send");
            MPI_Send(chunk, own_chunk_size, MPI_INT,
                     rank_of_process - step, 0,
                     MPI_COMM_WORLD);
            CALI_MARK_END("MPI_Send");
            CALI_MARK_END(comm_large);
            CALI_MARK_END(comm);

            break;
        }

        if (rank_of_process + step < number_of_process)
        {
            int received_chunk_size = (number_of_elements >= chunk_size * (rank_of_process + 2 * step))
                                          ? (chunk_size * step)
                                          : (number_of_elements - chunk_size * (rank_of_process + step));
            int *chunk_received;
            chunk_received = (int *)malloc(
                received_chunk_size * sizeof(int));
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_large);
            CALI_MARK_BEGIN("MPI_Recv");
            MPI_Recv(chunk_received, received_chunk_size,
                     MPI_INT, rank_of_process + step, 0,
                     MPI_COMM_WORLD, &status);
            CALI_MARK_END("MPI_Recv");
            CALI_MARK_END(comm_large);
            CALI_MARK_END(comm);


            compSmallTime = MPI_Wtime();

            CALI_MARK_BEGIN(comp);
            CALI_MARK_BEGIN(comp_small);
            data = merge(chunk, own_chunk_size,
                         chunk_received,
                         received_chunk_size);
            CALI_MARK_END(comp_small);
            CALI_MARK_END(comp);

            compSmallTime = MPI_Wtime() - compSmallTime;

            free(chunk);
            free(chunk_received);
            chunk = data;
            own_chunk_size = own_chunk_size + received_chunk_size;
        }
    }

    // Opening the other file as taken form input
    // and writing it to the file and giving it
    // as the output
    if (rank_of_process == 0)
    {
        correctTime = MPI_Wtime();

        CALI_MARK_BEGIN(correctness_check);
        // check if array is sorted or not
        if (isSorted(chunk, number_of_elements))
        {
            printf("Array is sorted!\n");
        }
        else
        {
            printf("Array is not sorted (womp womp)\n");
        }
        CALI_MARK_END(correctness_check);

        correctTime = MPI_Wtime() - correctTime;

        // Print sorted array
        if(printArray) {
            for (int i = 0; i < number_of_elements; i++)
            {
                printf("%d ", chunk[i]);
                if (i % 10 == 0) {
                    printf("\n");
                }
            }
            printf("\n");
        }

        totalTime = MPI_Wtime() - totalTime;
        printf("Time taken: %f\n", totalTime);
        printf("Data init time: %f\n", dataInitTime);
        printf("Barrier time: %f\n", barrierTime);
        printf("Comm small time: %f\n", commSmallTime);
        printf("Comm large time: %f\n", commLargeTime);
        printf("Comp small time: %f\n", compSmallTime);
        printf("Comp large time: %f\n", compLargeTime);
        printf("Correctness time: %f\n", correctTime);
    }

    totalTime = MPI_Wtime() - totalTime;

    // create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    adiak::init(NULL);
    adiak::launchdate();                                    // launch date of the job
    adiak::libraries();                                     // Libraries used
    adiak::cmdline();                                       // Command line used to launch the job
    adiak::clustername();                                   // Name of the cluster
    adiak::value("Algorithm", "QuickSort");                 // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("SizeOfDatatype", sizeof(int));            // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", number_of_elements); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // The number of elements in input dataset (1000)
    adiak::value("num_procs", number_of_process);      // The number of processors (MPI ranks)
    adiak::value("group_num", 23);
    adiak::value("implementation_source", "Online");

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
    MPI_Finalize();
    return 0;
}
