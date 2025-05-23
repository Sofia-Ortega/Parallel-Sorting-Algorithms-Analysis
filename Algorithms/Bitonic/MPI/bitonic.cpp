/**
 * -------------------- Adapted From -----------------------------------
 * Code: https://github.com/adrianlee/mpi-bitonic-sort/tree/master
 * Report: https://andreask.cs.illinois.edu/Teaching/HPCFall2012/Projects/yourii-report.pdf
 * Author: Adrian Lee
 * University: McGill University
 * Date: Dec 1, 2012
 * 
 *     References:
        Assignment 4 Supplement - Parallel Bitonic Sort Algorithm
        Sorting Algorithms - http://www-users.cs.umn.edu/~karypis/parbook/Algorithms/pchap9.pdf
        Bionic Sorting - http://www.thi.informatik.uni-frankfurt.de/~klauck/PDA07/Bitonic%20Sorting.pdf
*/

#include <stdio.h>      // Printf
#include <time.h>       // Timer
#include <math.h>       // Logarithm
#include <stdlib.h>     // Malloc
#include "mpi.h"        // MPI Library
#include "bitonic.h"
#include <string.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
// #include <adiak.h>

#define MASTER 0        // Who should do the final processing?
#define OUTPUT_NUM 10   // Number of elements to display in output

// Globals
// Not ideal for them to be here though
double timer_start;
double timer_end;
int process_rank;
int num_processes;
int * array;
int array_size;
int option;

// CALI variables 
const char* main_time = "main_time";
const char* data_init = "data_init";
const char* mpibarrier = "mpibarrier";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correct_check = "correct_check";

///////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////
int main(int argc, char * argv[]) {

    CALI_CXX_MARK_FUNCTION;
    
    // CALI_MARK_BEGIN("main_time");
    int i, j;

    // Initialization, get # of processes & this PID/rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    CALI_MARK_BEGIN("data_init");
    // Initialize Array for Storing Random Numbers
    array_size = atoi(argv[1]) / num_processes;
    array = (int *) malloc(array_size * sizeof(int));

    option = atoi(argv[2]);
    
    // Generate Random Numbers for Sorting (within each process)
    // Less overhead without MASTER sending random numbers to each slave
    srand(time(NULL));  // Needed for rand()
 
    // array_fill(array, array_size);
    if (option == 0) {
        // random
        for (int i = 0; i < array_size; i++) {
        array[i] = rand() % (atoi(argv[1]));
        }
    } 
    else if (option == 1) {
        // sorted
        for (int i = 0; i < array_size; i++) {
        array[i] = i;
        }
    }
    else if (option == 2) {
        // reverse sorted
        for (int i = 0; i < array_size; i++) {
        array[i] = array_size - i;
        }
    }
    else if (option == 3) {
        // perturb 1& of values
        for (int i = 0; i < array_size; i++) {
        array[i] = i;
        }

        int num_perturb = array_size / 100;
        for (int i = 0; i < num_perturb; i++) {
        int index = rand() % array_size;
        array[index] = rand() % array_size;
        }
    }

    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("mpibarrier");
    // Blocks until all processes have finished generating
    MPI_Barrier(MPI_COMM_WORLD);

    CALI_MARK_END("mpibarrier");

    
    // Begin Parallel Bitonic Sort Algorithm from Assignment Supplement

    // Cube Dimension
    int dimensions = (int)(log2(num_processes));

    // Start Timer before starting first sort operation (first iteration)
    if (process_rank == MASTER) {
        printf("Number of Processes spawned: %d\n", num_processes);
        timer_start = MPI_Wtime();
    }

    // Sequential Sort
    qsort(array, array_size, sizeof(int), ComparisonFunc);

    // CALI_MARK_BEGIN("comp_large");
    // Bitonic Sort follows
    for (i = 0; i < dimensions; i++) {
        for (j = i; j >= 0; j--) {
            // (window_id is even AND jth bit of process is 0)
            // OR (window_id is odd AND jth bit of process is 1)
            if (((process_rank >> (i + 1)) % 2 == 0 && (process_rank >> j) % 2 == 0) || ((process_rank >> (i + 1)) % 2 != 0 && (process_rank >> j) % 2 != 0)) {
                CompareLow(j);
            } else {
                CompareHigh(j);
            }
        }
    }
    // CALI_MARK_END("comp_large");

    // Blocks until all processes have finished sorting
    MPI_Barrier(MPI_COMM_WORLD);

    if (process_rank == MASTER) {
        timer_end = MPI_Wtime();

        CALI_MARK_BEGIN("correct_check");
        // Check if array is sorted
        for (i = 0; i < array_size - 1; i++) {
            if (array[i] > array[i + 1]) {
                printf("Array is not sorted!\n");
                
                break;
            }
        }
        CALI_MARK_END("correct_check");

        printf("Displaying sorted array (only 10 elements for quick verification)\n");

        // Print Sorting Results
        for (i = 0; i < array_size; i++) {
            if ((i % (array_size / OUTPUT_NUM)) == 0) {
                printf("%d ",array[i]);
            }
        }
        printf("\n\n");

        printf("Time Elapsed (Sec): %f\n", timer_end - timer_start);
    }

    // Reset the state of the heap from Malloc
    free(array);

    	// Create caliper ConfigManager object
	cali::ConfigManager mgr;
	mgr.start();

	adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "BitonicSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 8); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", atoi(argv[1])); // The number of elements in input dataset (1000)
	std::string inputType = "Random";
	if (option == 1) inputType = "Sorted";
	else if (option == 2) inputType = "ReverseSorted";
    else if (option == 3) inputType = "1% Perturbed";
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_processes); // The number of processors (MPI ranks)
    adiak::value("group_num", 23); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online/Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

   	// Flush Caliper output before finalizing MPI
   	mgr.stop();
   	mgr.flush();

    // Done
    MPI_Finalize();

    // CALI_MARK_END("main_time");
    
    return 0;
}

///////////////////////////////////////////////////
// Comparison Function
///////////////////////////////////////////////////
int ComparisonFunc(const void * a, const void * b) {
    
    
    return ( * (int *)a - * (int *)b );
}

///////////////////////////////////////////////////
// Compare Low
///////////////////////////////////////////////////
void CompareLow(int j) {
    
    int i, min;

    /* Sends the biggest of the list and receive the smallest of the list */
    CALI_MARK_BEGIN("comm");

    CALI_MARK_BEGIN("comm_small");
    // Send entire array to paired H Process
    // Exchange with a neighbor whose (d-bit binary) processor number differs only at the jth bit.
    int send_counter = 0;
    int *buffer_send = static_cast<int*>(malloc((array_size + 1) * sizeof(int)));
    
    MPI_Send(
        &array[array_size - 1],     // entire array
        1,                          // one data item
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD              // default comm.
    );
    CALI_MARK_END("comm_small");
    // Receive new min of sorted numbers
    int recv_counter;
    int *buffer_recieve = static_cast<int*>(malloc((array_size + 1) * sizeof(int)));
    MPI_Recv(
        &min,                       // buffer the message
        1,                          // one data item
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD,             // default comm.
        MPI_STATUS_IGNORE           // ignore info about message received
    );

    // Buffers all values which are greater than min send from H Process.
    for (i = 0; i < array_size; i++) {
        if (array[i] > min) {
            buffer_send[send_counter + 1] = array[i];
            send_counter++;
        } else {
            
            break;      // Important! Saves lots of cycles!
        }
    }

    CALI_MARK_BEGIN("comm_large");
    buffer_send[0] = send_counter;

    // send partition to paired H process
    MPI_Send(
        buffer_send,                // Send values that are greater than min
        send_counter,               // # of items sent
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD              // default comm.
    );

    // receive info from paired H process
    MPI_Recv(
        buffer_recieve,             // buffer the message
        array_size,                 // whole array
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD,             // default comm.
        MPI_STATUS_IGNORE           // ignore info about message received
    );

    CALI_MARK_END("comm_large");

    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");

    // Take received buffer of values from H Process which are smaller than current max
    for (i = 1; i < buffer_recieve[0] + 1; i++) {
        if (array[array_size - 1] < buffer_recieve[i]) {
            // Store value from message
            array[array_size - 1] = buffer_recieve[i];
        } else {
            
            break;      // Important! Saves lots of cycles!
        }
    }
    
    CALI_MARK_END("comp_small");

    CALI_MARK_BEGIN("comp_large");
    // Sequential Sort
    qsort(array, array_size, sizeof(int), ComparisonFunc);
    CALI_MARK_END("comp_large");

    CALI_MARK_END("comp");
    // Reset the state of the heap from Malloc
    free(buffer_send);
    free(buffer_recieve);
    
    return;
}


///////////////////////////////////////////////////
// Compare High
///////////////////////////////////////////////////
void CompareHigh(int j) {
    
    int i, max;

    // Receive max from L Process's entire array
    int recv_counter;
    int *buffer_recieve = static_cast<int*>(malloc((array_size + 1) * sizeof(int)));
    MPI_Recv(
        &max,                       // buffer max value
        1,                          // one item
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD,             // default comm.
        MPI_STATUS_IGNORE           // ignore info about message received
    );

    // Send min to L Process of current process's array
    int send_counter = 0;
    int *buffer_send = static_cast<int*>(malloc((array_size + 1) * sizeof(int)));
    MPI_Send(
        &array[0],                  // send min
        1,                          // one item
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD              // default comm.
    );

    // Buffer a list of values which are smaller than max value
    for (i = 0; i < array_size; i++) {
        if (array[i] < max) {
            buffer_send[send_counter + 1] = array[i];
            send_counter++;
        } else {
            
            break;      // Important! Saves lots of cycles!
        }
    }

    // Receive blocks greater than min from paired slave
    MPI_Recv(
        buffer_recieve,             // buffer message
        array_size,                 // whole array
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD,             // default comm.
        MPI_STATUS_IGNORE           // ignore info about message receiveds
    );
    recv_counter = buffer_recieve[0];

    // send partition to paired slave
    buffer_send[0] = send_counter;
    MPI_Send(
        buffer_send,                // all items smaller than max value
        send_counter,               // # of values smaller than max
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD              // default comm.
    );

    // Take received buffer of values from L Process which are greater than current min
    for (i = 1; i < recv_counter + 1; i++) {
        if (buffer_recieve[i] > array[0]) {
            // Store value from message
            array[0] = buffer_recieve[i];
        } else {
            
            break;      // Important! Saves lots of cycles!
        }
    }

    // Sequential Sort
    qsort(array, array_size, sizeof(int), ComparisonFunc);

    // Reset the state of the heap from Malloc
    free(buffer_send);
    free(buffer_recieve);
    
    
    return;
}