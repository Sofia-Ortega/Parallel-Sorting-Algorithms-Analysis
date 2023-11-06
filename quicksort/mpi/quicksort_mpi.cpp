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

// Simple swap function
void swap(int *arr, int i, int j)
{
    int temp = arr[j];
    arr[j] = arr[i];
    arr[i] = temp;
}

// Quick sort function
void quicksort(int *arr, int start, int end)
{

    int pivot, index;

    // base case
    if (end <= 1)
    {
        return;
    }

    // select pivot
    pivot = arr[start + end / 2];
    swap(arr, start, start + end / 2);
    index = start;

    for (int i = start + 1; i < start + end; i++)
    {
        if (arr[i] < pivot)
        {
            ++index;
            swap(arr, i, ++index);
        }
    }

    swap(arr, start, index);

    // Recursively sort
    quicksort(arr, start, index - start);
    quicksort(arr, index + 1, end - (index - start) - 1);
}

// once you have two sorted arrays - need to merge them
int *merge(int *arr1, int n1, int *arr2, int n2)
{
    int *result = (int *)malloc((n1 + n2) * sizeof(int)); // dedicate space of size arr1 and 2
    int i = 0, j = 0, k = 0;

    for (k = 0; k < (n1 + n2); k++)
    {
        if (i >= n1)
        {
            result[k] = arr2[j];
            j++;
        }
        else if (j >= n1)
        {
            result[k] = arr1[i];
            i++;
        }
        else if (arr1[i] < arr2[j])
        {
            result[k] = arr1[i];
            i++;
        }
        else
        {
            result[k] = arr2[j];
            j++;
        }
    }

    return result;
}

// driver code and MPI functionality
int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;

    // Read in CLI arguments
    int number_of_elements = atoi(argv[1]); // size of array
    int size = atoi(argv[2]);               // 0 for small,

    int *data = NULL;
    int chunk_size, own_chunk_size;
    int *chunk;
    double dataInitTime, barrierTime, commTime, compTime, correctTime, totalTime;
    MPI_Status status;

    totalTime = MPI_Wtime();
    CALI_MARK_BEGIN("totalTime")

    // Initialize MPI
    int process_id, process_count;
    MPI_Init(&argc, &argv);
    MPI_COMM_size(MPI_COMM_WORLD, &process_count);
    MPI_COMM_rank(MPI_COMM_WORLD, &process_id);
    // data_init
    dataInitTime = MPI_Wtime();
    CALI_MARK_BEGIN("dataInitTime")
    if (process_id == 0)
    {
        genValues(process_id, process_count, number_of_elements, false, data, 0);
    }
    CALI_MARK_END("dataInitTime")
    dataInitTime = MPI_Wtime() - dataInitTime;

    // comm_small region

    // block all child processes until master process
    // has finished generating data
    barrierTime = MPI_Wtime();
    CALI_MARK_BEGIN("barrierTime")
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("barrierTime")
    barrierTime = MPI_Wtime() - barrierTime;

    // Broadcast the number of elements
    commTime = MPI_Wtime();
    CALI_MARK_BEGIN("commTime")
    MPI_Bcast(&number_of_elements, 1, MPI_INT, 0,
              MPI_COMM_WORLD);

    // Computing chunk size
    chunk_size = (number_of_elements % process_count == 0)
                     ? (number_of_elements / process_count)
                     : (number_of_elements / process_count - 1);

    // Allocate memory for chunk
    chunk = (int *)malloc(chunk_size * sizeof(int));

    // Scatter the data
    MPI_Scatter(data, chunk_size, MPI_INT, chunk,
                chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    free(data);
    data = NULL;
    CALI_MARK_END("commTime")

    // Compute own chunk size
    compTime = MPI_Wtime();
    CALI_MARK_BEGIN("compTime")
    own_chunk_size = (process_id == process_count - 1)
                         ? (chunk_size + number_of_elements % process_count)
                         : chunk_size;

    // Sort the data for every chunk called by process
    quicksort(chunk, 0, own_chunk_size);
    for (step = 1; step < process_count; step *= 2)
    {
        if (process_id % (2 * step) != 0)
        {
            MPI_Send(chunk, own_chunk_size, MPI_INT, process_id - step, 0, MPI_COMM_WORLD);
            break;
        }
        if (process_id + step < process_count)
        {
            int received_chunk_size = (number_of_elements >= chunk_size * (process_id + 2 * step))
                                          ? (chunk_size * step)
                                          : (number_of_elements - chunk_size * (process_id + step));
            int *chunk_received;
            chunk_received = (int *)malloc(
                received_chunk_size * sizeof(int));
            MPI_Recv(chunk_received, received_chunk_size,
                     MPI_INT, process_id + step, 0,
                     MPI_COMM_WORLD, &status);

            data = merge(chunk, own_chunk_size,
                         chunk_received,
                         received_chunk_size);

            free(chunk);
            free(chunk_received);
            chunk = data;
            own_chunk_size = own_chunk_size + received_chunk_size;
        }
    }

    CALI_MARK_END("compTime")
    compTime = MPI_Wtime() - compTime;

    correctTime = MPI_Wtime();
    CALI_MARK_BEGIN("correctTime")

    if (process_id == 0)
    {
        // Opening the file
        file = fopen(argv[2], "w");

        if (file == NULL)
        {
            printf("Error in opening file... \n");
            exit(-1);
        }

        // Printing total number of elements
        // in the file
        fprintf(
            file,
            "Total number of Elements in the array : %d\n",
            own_chunk_size);

        // Printing the value of array in the file
        for (int i = 0; i < own_chunk_size; i++)
        {
            fprintf(file, "%d ", chunk[i]);
        }

        // Closing the file
        fclose(file);

        printf("\n\n\n\nResult printed in output.txt file "
               "and shown below: \n");

        // For Printing in the terminal
        printf("Total number of Elements given as input : "
               "%d\n",
               number_of_elements);
        printf("Sorted array is: \n");

        for (int i = 0; i < number_of_elements; i++)
        {
            printf("%d ", chunk[i]);
        }

        printf(
            "\n\nQuicksort %d ints on %d procs: %f secs\n",
            number_of_elements, process_count,
            time_taken);
    }

    CALI_MARK_END("totalTime")
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
    adiak::value("number_of_elements", number_of_elements); // The number of elements in input dataset (1000)
    adiak::value("process_counts", process_count);          // The number of processors (MPI ranks)


    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
    MPI_Finalize();
}
