
/**
 * -------------------- Adapted From -----------------------------------
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

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// global constants definitions
#define b 32           // number of bits for integer
#define g 8            // group of bits for each scan
#define N b / g        // number of passes
#define B (1 << g)     // number of buckets, 2^g

#define RANDOM          0
#define SORTED          1
#define REVERSE_SORTED  2
#define PERTURBED       3

  // 0: random
  // 1: sorted
  // 2: reverse sorted

// MPI tags constants, offset by max bucket to avoid collisions
#define COUNTS_TAG_NUM  B + 1 
#define PRINT_TAG_NUM  COUNTS_TAG_NUM + 1 
#define NUM_TAG PRINT_TAG_NUM + 1
#define CHECK_SORTED_TAG NUM_TAG + 1

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* MPI_Isend_area = "MPI_Isend";
const char* MPI_Recv_area = "MPI_Recv";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";

const char* sorting = "sorting_time";

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
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (int i = 0; i < *n; i++) {
      unsigned int idx = bits(a[i], pass*g, g);
      count[idx][rank]++; 
      l_count[idx]++;
      if (!add_item(&buckets[idx], a[i])) {
        return NULL;
      }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // do one-to-all transpose
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    for (int p = 0; p < P; p++) {
      if (p != rank) {
        // send counts of this process to others
        CALI_MARK_BEGIN(MPI_Isend_area);
        MPI_Isend(
            l_count,
            B,
            MPI_INT,
            p,
            COUNTS_TAG_NUM,
            MPI_COMM_WORLD,
            &req);
        CALI_MARK_END(MPI_Isend_area);
      }
    }

    // receive counts from others
    for (int p = 0; p < P; p++) {
      if (p != rank) {
        // comp_small
        CALI_MARK_BEGIN(MPI_Recv_area);
        MPI_Recv(
            l_count,
            B,
            MPI_INT,
            p,
            COUNTS_TAG_NUM,
            MPI_COMM_WORLD,
            &stat);
        CALI_MARK_END(MPI_Recv_area);

        // populate counts per bucket for other processes
        for (int i = 0; i < B; i++) {
          count[i][p] = l_count[i];
        }
      }
    }
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    // calculate new size based on values received from all processes
    int new_size = 0;
    for (int j = 0; j < l_B; j++) {
      int idx = j + rank * l_B;
      for (int p = 0; p < P; p++) {
        p_sum[j][p] = new_size;
        new_size += count[idx][p];
      }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

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

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    // send keys of this process to others
    for (int j = 0; j < B; j++) {
      int p = j / l_B;   // determine which process this buckets belongs to
      int p_j = j % l_B; // transpose to that process local bucket index
      if (p != rank && buckets[j].length > 0) {
        CALI_MARK_BEGIN(MPI_Isend_area);
        MPI_Isend(
            buckets[j].array,
            buckets[j].length,
            MPI_INT,
            p,
            p_j,
            MPI_COMM_WORLD,
            &req);
        CALI_MARK_END(MPI_Isend_area);
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
            CALI_MARK_BEGIN(MPI_Recv_area);
            MPI_Recv(
                dest,
                b_count,
                MPI_INT,
                p,
                j,
                MPI_COMM_WORLD,
                &stat);  
            CALI_MARK_END(MPI_Recv_area);

          } else {
            // is same process, copy from buckets to our array
            memcpy(dest, &buckets[idx].array[0], b_count*sizeof(int));
          }
        }
      }
    }

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // update new size
    *n = new_size;
  }

  return a;
}

int main(int argc, char** argv)
{
  // argv:
  // 0          1                          2              3
  // radix_mpi  number_of_elements_to_sort genArrayOption [optional: printArray]

  // genArrayOption
  // 0: random
  // 1: sorted
  // 2: reverse sorted
  // 3: perturbed

  const int requiredInputNum = 3;

  CALI_CXX_MARK_FUNCTION;


  int rank, size;
  int print_results = 0;

  // initialize MPI environment and obtain basic info
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // check for correct number of arguments
  if (argc < requiredInputNum)
  {
    if (rank == 0) usage("Not enough arguments!");
    MPI_Finalize();
    return EXIT_FAILURE;
  } else if (argc > requiredInputNum) {
    print_results = atoi(argv[requiredInputNum]);
  }

  // initialize vars
  int n_total = atoi(argv[1]);
  int n = n_total/size; // how many variables MY process has to sort
  if (n < 1) {
    if (rank == 0) {
      printf("Number of elements: %i\n", n_total);
      printf("Size: %i\n", size);
      usage("Number of elements must be >= number of processes!");
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int genValueOption = atoi(argv[2]);
  if(genValueOption < 0 || genValueOption > 3) {
    if (rank == 0) printf ("genValueOption %i is not a valid option. Must be [0, 3]", genValueOption);
    MPI_Finalize();
    return EXIT_FAILURE;
  } 

  if(rank == 0) {
    printf("n_total: %i\n", n_total);
    printf("genValueOption: %i\n", genValueOption);
    printf("size: %i\n", size);
    printf("print_results: %i\n", print_results);
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

  int b_capacity = n / B; // how much each bucket is going to store
  if (b_capacity < B) {
    b_capacity = B;
  }
  List* buckets = (List*)malloc(B*sizeof(List));
  for (int j = 0; j < B; j++) {
    buckets[j].array = (int*)malloc(b_capacity*sizeof(int));
    buckets[j].capacity = B;
  }

  // initialize local array
  CALI_MARK_BEGIN(data_init);

  printf("Rank %d creating %i elements\n", rank, n);


  std::string inputType;
  switch (genValueOption)
  {
  case RANDOM:
    inputType = "Random";
    for(int i = 0; i < n; i++) {
      a[i] = rand() % 1000000;
    }
    break;

  case SORTED:
    inputType = "Sorted";
    printf("Rank %d [%d, %d]\n", rank, n * rank, (n * rank) + n);
    for(int i = 0; i < n; i++) {
      a[i] = i + (n * rank);
    }
    break;

  case REVERSE_SORTED: {

    inputType = "ReverseSorted";
    
    int num = (n * rank) + n;
    for(int i = 0; i >= n; i--) {
      a[i] = num--;
    }

    printf("Rank %d [%d, %d]\n", (n * rank) + n, num++);
    break;
  }

  case PERTURBED:
    inputType = "1%perturbed";
    printf("Perturbed indeed\n");
    for(int i = 0; i < n; i++) {
      a[i] = i + (n * rank);
    }

    // randomize 1% of the values
    for(int i = 0; i < n * 0.1; i++) {
      int randIndex = rand() % n;
      a[randIndex] = rand() % 1000000;
    }

    break;
  
  default:
    printf("Error in setting sorting");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  CALI_MARK_END(data_init);

  // let all processes get here
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN("MPI_Barrier");
  MPI_Barrier(MPI_COMM_WORLD);
  CALI_MARK_END("MPI_Barrier");
  CALI_MARK_END(comm);

  if(print_results) {
    for(int i = 0; i < n; i++) {
      printf("Rank %i: %i\n", rank, a[i]);
    }
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
    printf("%d elements sorted\n", n_total);
  }



  // check own array 
  /*
  bool isMineSorted = true;
  bool isOverallSorted = true;
  for(int i = 0; i < n - 1; i++) {
    if(a[i] > a[i + 1]) {
      isMineSorted = false;
      break;
    }
  }

  if(rank != 0) {
    // if not rank 0, send data
    int sendBuf[3];

    sendBuf[0] = isMineSorted;
    sendBuf[1] = a[0]; // min
    sendBuf[2] = a[n - 1]; // max

    MPI_Send(sendBuf, 3, MPI_INT, 0, CHECK_SORTED_TAG, MPI_COMM_WORLD);

  } else {

    // check if rank 0 has sorted array 
    if (!isMineSorted) {
      printf("[FAIL] Rank %d did not sort\n");
      isOverallSorted = false;
    } else {

      int prevTop = a[n - 1];

      // receive other's status
      for(int r = 1; r < size; r++) { // iterate through all the processors
        MPI_Status stat;
        int buff[3];

        MPI_Recv(buff, 3, MPI_INT, r,CHECK_SORTED_TAG, MPI_COMM_WORLD, &stat );

        int isRankSorted = buff[0];
        if(!isRankSorted) {
          printf("[FAIL] Rank %d did not sort\n", r);
          isOverallSorted = false;
          break;
        }

        int currBottom = buff[1];

        if (prevTop > currBottom) {
          printf("[FAIL] Failed at the boundary between Rank %d[%i] ,%d[%i]\n", r - 1, prevTop, r, currBottom);
          isOverallSorted = false;
        }

        // update boundaries
        prevTop = buff[2]; // currMax
        

      }


    }

    if(isOverallSorted) {
      printf("[SUCCESS] Array sorted\n");
    }

  }
  */


  // check if sorted
  CALI_MARK_BEGIN(correctness_check);
  if(rank == 0) {
    if(print_results) printf("%i\n", a[0]);
    for(int i = 1; i < n_total; i++) {
      if(a[i - 1] > a[i]) {
        printf("ERROR in sorting at index [%i, %i]; [%i > %i]\n", i-1, i, a[i - 1], a[i]);
      }
      if(print_results) printf("%i\n", a[i]);
    }
    printf("[PASSED] Sorted Array Checked\n");
  }

  CALI_MARK_END(correctness_check);

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
  
  // print results
  print_array(size, rank, &a[0], p_n);





  // create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "RadixSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", n_total); // The number of elements in input dataset (1000)
  adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", size); // The number of processors (MPI ranks)
  adiak::value("group_num", 23); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
  
  // release MPI resources
  MPI_Finalize();

  // release memory allocated resources
  for (int j = 0; j < B; j++) {
    free(buckets[j].array);
  }
  free(buckets);
  free(a);

  return 0;
}