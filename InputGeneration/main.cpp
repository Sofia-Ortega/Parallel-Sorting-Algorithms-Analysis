#include "inputgen.h"

int main (int argc, char *argv[])
{
	CALI_CXX_MARK_FUNCTION;

	char * createSortedArray = "sorted_array_time";

	int numberOfVals = atoi(argv[1]);
	bool isDoubles = (atoi(argv[2]) == 1);
	int option = atoi(argv[3]);

	int taskid;
	int numprocs;
    
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	//void *values;
	double *valuesDouble;
	int *valuesInt = nullptr;
	
	if (taskid == ROOT)
	{
		if (isDoubles) 
	 	{	
			valuesDouble = new double[numberOfVals];
		}
		else 
		{	

			valuesInt = new int[numberOfVals];
		}
	}
	
	double genValuesTime = MPI_Wtime();
	if (isDoubles) genValues(taskid, numprocs, numberOfVals, isDoubles, valuesDouble, option);
	else genValues(taskid, numprocs, numberOfVals, isDoubles, valuesInt, option);
	genValuesTime = MPI_Wtime() - genValuesTime;

	if (taskid == ROOT)
	{

		//printVals<int>(valuesInt, numberOfVals);
		if (isDoubles)
		{
			delete[] valuesDouble;
			double *valsSeqRand = new double[numberOfVals];
			double randArrayTimeSeq = MPI_Wtime();
			fillValsRandSequential(valsSeqRand, numberOfVals, numprocs*3);
			randArrayTimeSeq = MPI_Wtime() - randArrayTimeSeq;

			printf("genValues Sequence Time: %.3f\n", randArrayTimeSeq);

			delete[] valsSeqRand;
		}
		else
		{

			delete[] valuesInt;
			int *valsSeqRand = new int[numberOfVals];
			double randArrayTimeSeq = MPI_Wtime();
			fillValsRandSequential(valsSeqRand, numberOfVals, numprocs*3);
			randArrayTimeSeq = MPI_Wtime() - randArrayTimeSeq;

			printf("genValues Sequence Time: %.3f\n", randArrayTimeSeq);

			delete[] valsSeqRand;
		}
		
		printf("genValues Parallel Time: %.3f\n", genValuesTime);

	}

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

   	MPI_Finalize();
}
