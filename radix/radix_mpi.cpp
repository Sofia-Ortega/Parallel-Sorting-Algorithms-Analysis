
/**
 * -------------------- SOURCE -----------------------------------
 * Code: https://github.com/ym720/p_radix_sort_mpi/blob/master/p_radix_sort_mpi/p_radix_sort.c
 * Report: https://andreask.cs.illinois.edu/Teaching/HPCFall2012/Projects/yourii-report.pdf
 * Author: Yourii Martiak
 * University: New York University
 * Date: December 20, 2012
 * 
*/


/**
 * Parallel implementation of radix sort. The list to be sorted is split
 * across multiple MPI processes and each sub-list is sorted during each
 * pass as in straight radix sort. 
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "timing.h"
#include "inputgen.h"

// global constants definitions
#define b 32           // number of bits for integer
#define g 8            // group of bits for each scan
#define N b / g        // number of passes
#define B (1 << g)     // number of buckets, 2^g

// MPI tags constants, offset by max bucket to avoid collisions
#define COUNTS_TAG_NUM  B + 1 
#define PRINT_TAG_NUM  COUNTS_TAG_NUM + 1 
#define NUM_TAG PRINT_TAG_NUM + 1
#define CHECK_SORTED NUM_TAG + 1

// structure encapsulating buckets with arrays of elements
typedef struct list List;
struct list {
  int* array;
  size_t length;
  size_t capacity;
};

// add item to a dynamic array encapsulated in a structure
int add_item(List* list, int item) {
  if (list->length >= list->capacity) {
    size_t new_capacity = list->capacity*2;
    int* temp = (int*)realloc(list->array, new_capacity*sizeof(int));
    if (!temp) {
      printf("ERROR: Could not realloc for size %d!\n", (int) new_capacity); 
      return 0;
    }
    list->array = temp;
    list->capacity = new_capacity;
  }

  list->array[list->length++] = item;

  return 1;
}

void usage(char* message) {
  fprintf(stderr, "Incorrect usage! %s\n", message);
  fprintf(stderr, "Usage: mpiexec -n [processes] p_radix_sort [f] [n] [r]\n");
  fprintf(stderr, "  [processes] - number of processes to use\n");
  fprintf(stderr, "  [f] - input file to be sorted\n");
  fprintf(stderr, "  [n] - number of elements in the file\n");
  fprintf(stderr, "  [r] - print sorted results 0/1, 0 by default\n");
}

// print resulting array while gathering information from all processes
/**
 * P: total number of sorted values 
 * rank: MPI rank
 * a: array elements in your own process
 * n: array containing the number of sorted elements in each process
*/
void print_array(const int P, const int rank, int *a, int *n) {
  if (rank == 0) {

    // print array for rank 0 first
    printf("\nProcess 0 with %i elements\n", n[rank]);
    for (int i = 0; i < n[rank]; i++) {
      printf("%d\n", a[i]);
    } 
    // then receive and print from others
    for (int p = 1; p < P; p++) {
      MPI_Status stat;
      int a_size = n[p];
      int buff[a_size];
      MPI_Recv(buff, a_size, MPI_INT, p, PRINT_TAG_NUM, MPI_COMM_WORLD, &stat);
      printf("\nProcess %i with %i elements\n", p, a_size);
      for (int i = 0; i < a_size; i++) {
        printf("%d\n", buff[i]);
      } 
    }
  } else {
    // if not rank 0, send your data to other processes
    MPI_Send(a, n[rank], MPI_INT, 0, PRINT_TAG_NUM, MPI_COMM_WORLD); 
  }
}


// Initialize array with numbers read from a file
// initialize array 
int init_array(char* file, const int begin, const int n, int *a) {


  // open file in read-only mode and check for errors
  FILE *file_ptr;
  file_ptr = fopen(file, "r");
  if (file_ptr == NULL) {
    return EXIT_FAILURE;
  }

  // read n numbers from a file into array a starting at begin position
  int skip;

  // first skip to the begin position
  for (int i = 0; i < begin; i++) {
    int s = fscanf(file_ptr, "%d", &skip); 
  }

  // then read numbers into array a
  for (int i = 0; i < n; i++) {
    int s = fscanf(file_ptr, "%d", &a[i]);
  }

  return EXIT_SUCCESS;
}

// Compute j bits which appear k bits from the right in x
// Ex. to obtain rightmost bit of x call bits(x, 0, 1)
unsigned bits(unsigned x, int k, int j) {
  return (x >> k) & ~(~0 << j);
}

// Radix sort elements while communicating between other MPI processes
// a - array of elements to be sorted
// buckets - array of buckets, each bucket pointing to array of elements
// P - total number of MPI processes
// rank - rank of this MPI process
// n - number of elements to be sorted
int* radix_sort(int *a, List* buckets, const int P, const int rank, int * n) {
  int count[B][P];   // array of counts per bucket for all processes
  int l_count[B];    // array of local process counts per bucket
  int l_B = B / P;   // number of local buckets per process
  int p_sum[l_B][P]; // array of prefix sums

  // MPI request and status
  MPI_Request req;
  MPI_Status stat;

  for (int pass = 0; pass < N; pass++) {          // each pass

    // init counts arrays
    for (int j = 0; j < B; j++) {
      count[j][rank] = 0;
      l_count[j] = 0;
      buckets[j].length = 0;
    } 

    // count items per bucket
    for (int i = 0; i < *n; i++) {
      unsigned int idx = bits(a[i], pass*g, g);
      count[idx][rank]++; 
      l_count[idx]++;
      if (!add_item(&buckets[idx], a[i])) {
        return NULL;
      }
    }

    // do one-to-all transpose
    for (int p = 0; p < P; p++) {
      if (p != rank) {
        // send counts of this process to others
        MPI_Isend(
            l_count,
            B,
            MPI_INT,
            p,
            COUNTS_TAG_NUM,
            MPI_COMM_WORLD,
            &req);
      }
    }

    // receive counts from others
    for (int p = 0; p < P; p++) {
      if (p != rank) {
        MPI_Recv(
            l_count,
            B,
            MPI_INT,
            p,
            COUNTS_TAG_NUM,
            MPI_COMM_WORLD,
            &stat);

        // populate counts per bucket for other processes
        for (int i = 0; i < B; i++) {
          count[i][p] = l_count[i];
        }
      }
    }

    // calculate new size based on values received from all processes
    int new_size = 0;
    for (int j = 0; j < l_B; j++) {
      int idx = j + rank * l_B;
      for (int p = 0; p < P; p++) {
        p_sum[j][p] = new_size;
        new_size += count[idx][p];
      }
    }

    // reallocate array if newly calculated size is larger
    if (new_size > *n) {
      int* temp = (int*)realloc(a, new_size*sizeof(int));
      if (!a) {
        if (rank == 0) {
          printf("ERROR: Could not realloc for size %d!\n", new_size); 
        }
        return NULL;
      }
      // reassign pointer back to original
      a = temp;
    }

    // send keys of this process to others
    for (int j = 0; j < B; j++) {
      int p = j / l_B;   // determine which process this buckets belongs to
      int p_j = j % l_B; // transpose to that process local bucket index
      if (p != rank && buckets[j].length > 0) {
        MPI_Isend(
            buckets[j].array,
            buckets[j].length,
            MPI_INT,
            p,
            p_j,
            MPI_COMM_WORLD,
            &req);
      }
    }

    // receive keys from other processes
    for (int j = 0; j < l_B; j++) {
      // transpose from local to global index 
      int idx = j + rank * l_B; 
      for (int p = 0; p < P; p++) {

        // get bucket count
        int b_count = count[idx][p]; 
        if (b_count > 0) {

          // point to an index in array where to insert received keys
          int *dest = &a[p_sum[j][p]]; 
          if (rank != p) {
            MPI_Recv(
                dest,
                b_count,
                MPI_INT,
                p,
                j,
                MPI_COMM_WORLD,
                &stat);  

          } else {
            // is same process, copy from buckets to our array
            memcpy(dest, &buckets[idx].array[0], b_count*sizeof(int));
          }
        }
      }
    }

    // update new size
    *n = new_size;
  }

  return a;
}

int main(int argc, char** argv)
{
  // argv:
  // 0         1               2
  // radix_mpi input_file_name number_of_elements_to_sort

  int rank, size;
  int print_results = 0;

  // initialize MPI environment and obtain basic info
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // check for correct number of arguments
  if (argc < 3)
  {
    if (rank == 0) usage("Not enough arguments!");
    MPI_Finalize();
    return EXIT_FAILURE;
  } else if (argc > 3) {
    print_results = atoi(argv[3]);
  }

  // initialize vars and allocate memory
  int n_total = atoi(argv[2]);
  int n = n_total/size;
  if (n < 1) {
    if (rank == 0) {
      printf("Number of elements: %i", n_total);
      printf("Size: %i", size);
      usage("Number of elements must be >= number of processes!");
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  if(rank == 0) {
    printf("n_total: %i\n", n_total);
    printf("size: %i\n", size);
  }

  int remainder = B % size;   // in case number of buckets is not divisible
  if (remainder > 0) {
    if (rank == 0) {
      usage("Number of buckets must be divisible by number of processes\n");
    } 
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // allocate memory and initialize buckets
  // if n is not divisible by size, make the last process handle the reamainder
  if (rank == size-1) {
    int remainder = n_total % size;
    if (remainder > 0) {
      n += remainder;
    }
  }

  const int s = n * rank;
  int* a = (int*)malloc(sizeof(int) * n);

  int b_capacity = n / B;
  if (b_capacity < B) {
    b_capacity = B;
  }
  List* buckets = (List*)malloc(B*sizeof(List));
  for (int j = 0; j < B; j++) {
    buckets[j].array = (int*)malloc(b_capacity*sizeof(int));
    buckets[j].capacity = B;
  }

  // initialize local array
  fillValsRandParallel(a, n, 10 + rank);

  // let all processes get here
  MPI_Barrier(MPI_COMM_WORLD);

  for(int i = 0; i < n; i++) {
    printf("Rank %i: %i\n", rank, a[i]);
  }
  
  // take a timestamp before the sort starts
  timestamp_type time1, time2;
  if (rank == 0) {
    get_timestamp(&time1);
  }

  // then run the sorting algorithm
  a = radix_sort(&a[0], buckets, size, rank, &n);

  if (a == NULL) {
    printf("ERROR: Sort failed, exiting ...\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // wait for all processes to finish before printing results 
  MPI_Barrier(MPI_COMM_WORLD);

  // take a timestamp after the process finished sorting
  if (rank == 0) {
    get_timestamp(&time2);

    // calculate fish updates per second
    double elapsed = timestamp_diff_in_seconds(time1,time2);
    printf("%f s\n", elapsed);
    printf("%d elements sorted\n", n_total);
    printf("%f elements/s\n", n_total / elapsed);

  }

  // store number of items per each process after the sort
  int* p_n = (int*)malloc(size*sizeof(int));

  // first store our own number
  p_n[rank] = n;

  // communicate number of items among other processes
  MPI_Request req;
  MPI_Status stat;

  for (int i = 0; i < size; i++) {
    if (i != rank) {
      MPI_Isend(
          &n,
          1,
          MPI_INT,
          i,
          NUM_TAG,
          MPI_COMM_WORLD,
          &req);
    }
  }

  for (int i = 0; i < size; i++) {
    if (i != rank) {
      MPI_Recv(
         &p_n[i],
         1,
         MPI_INT,
         i,
         NUM_TAG,
         MPI_COMM_WORLD,
         &stat);
    }
  }


  // check if sorted
  if(rank == 0) {
    printf("%i\n", a[0]);
    for(int i = 1; i < n_total; i++) {
      if(a[i - 1] > a[i]) {
        printf("ERROR in sorting at index [%i, %i]; [%i > %i]\n", i-1, i, a[i - 1], a[i]);
      }
      printf("%i\n", a[i]);
    }
    printf("Sorted Array Checked\n");
  }

  
  // print results
  if (print_results || true) {
    print_array(size, rank, &a[0], p_n);
  }

  // release MPI resources
  MPI_Finalize();

  // release memory allocated resources
  for (int j = 0; j < B; j++) {
    free(buckets[j].array);
  }
  free(buckets);
  free(a);
  free(p_n);

  return 0;
}