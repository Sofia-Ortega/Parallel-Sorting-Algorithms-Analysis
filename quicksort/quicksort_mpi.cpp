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
    int number_of_elements;
    int *data = NULL;
    int chunk_size, own_chunk_size;
    int *chunk;
    FILE *file = NULL;
    double time_taken;
    MPI_Status status;

    if (argc != 3)
    {
        printf("Desired number of arguments are not their "
               "in argv....\n");
        printf("2 files required first one input and "
               "second one output....\n");
        exit(-1);
    }

    // Initialize MPI
    int process_id, process_count;
    MPI_Init(&argc, &argv);

    // Get the number of processes
    if (rc != MPI_SUCCESS)
    {
        printf("Error in creating MPI "
               "program.\n "
               "Terminating......\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_COMM_size(MPI_COMM_WORLD, &process_count);
    MPI_COMM_rank(MPI_COMM_WORLD, &process_id);

    // Master process
    if (process_id == 0)
    {
        // Opening the file
        file = fopen(argv[1], "r");

        // Printing Error message if any
        if (file == NULL)
        {
            printf("Error in opening file\n");
            exit(-1);
        }

        // Reading number of Elements in file ...
        // First Value in file is number of Elements
        printf(
            "Reading number of Elements From file ....\n");
        fscanf(file, "%d", &number_of_elements);
        printf("Number of Elements in the file is %d \n",
               number_of_elements);

        // Computing chunk size
        chunk_size = (number_of_elements % process_count == 0)
                         ? (number_of_elements / process_count)
                         : (number_of_elements / process_count - 1);

        data = (int *)malloc(process_count * chunk_size * sizeof(int));

        // Reading the rest elements in which
        // operation is being performed
        printf("Reading the array from the file.......\n");
        for (int i = 0; i < number_of_elements; i++)
        {
            fscanf(file, "%d", &data[i]);
        }

        // Padding data with zero
        for (int i = number_of_elements;
             i < process_count * chunk_size; i++)
        {
            data[i] = 0;
        }

        // Printing the array read from file
        printf("Elements in the array is : \n");
        for (int i = 0; i < number_of_elements; i++)
        {
            printf("%d ", data[i]);
        }

        printf("\n");

        fclose(file);
        file = NULL;
    }

    // block all child processes until master process
    // has finished reading the file
    MPI_Barrier(MPI_COMM_WORLD);

    // Start timer
    time_taken = MPI_Wtime();

    // Broadcast the number of elements
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

    // Compute own chunk size
    own_chunk_size = (process_id == process_count - 1)
                         ? (chunk_size + number_of_elements % process_count)
                         : chunk_size;

    // Sort the data for every chunk called by process
    quicksort(chunk, 0, own_chunk_size);
    for (step = 1; step < process_count; step *= 2) {
        if (process_id % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT, process_id - step, 0, MPI_COMM_WORLD);
            break;
        }
        if (process_id + step < process_count) {
            int received_chunk_size
                = (number_of_elements
                   >= chunk_size
                          * (process_id + 2 * step))
                      ? (chunk_size * step)
                      : (number_of_elements
                         - chunk_size
                               * (process_id + step));
            int* chunk_received;
            chunk_received = (int*)malloc(
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
            own_chunk_size
                = own_chunk_size + received_chunk_size;
        }
    }

    // Stop the timer
    time_taken += MPI_Wtime();

    if (process_id == 0)
    {
        // Opening the file
        file = fopen(argv[2], "w");
 
        if (file == NULL) {
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
        for (int i = 0; i < own_chunk_size; i++) {
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
 
        for (int i = 0; i < number_of_elements; i++) {
            printf("%d ", chunk[i]);
        }
 
        printf(
            "\n\nQuicksort %d ints on %d procs: %f secs\n",
            number_of_elements, process_count,
            time_taken);
    }
}
