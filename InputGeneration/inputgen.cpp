/******************************************************************************
* FILE: inputgen.cpp
* DESCRIPTION:  
*   Generates large arrays of numbers in parallel to be used by sorting algos.
* AUTHOR: Will Thompson
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int main (int argc, char *argv[])
{
	CALI_CXX_MARK_FUNCTION;

	int taskid;
	int numtasks;
    
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	// Create caliper ConfigManager object
	cali::ConfigManager mgr;
	mgr.start();

   	adiak::init(NULL);
   	adiak::user();
   	adiak::launchdate();
   	adiak::libraries();
   	adiak::cmdline();
   	adiak::clustername();
   	adiak::value("num_procs", numtasks);
   	adiak::value("program_name", "Input Generation");

   	// Flush Caliper output before finalizing MPI
   	mgr.stop();
   	mgr.flush();

   	MPI_Finalize();
}
