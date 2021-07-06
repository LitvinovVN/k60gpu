#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <thread>
#include <vector>
#include <chrono> // chrono::system_clock
#include <ctime>   // localtime
#include <iomanip> // put_time

#include "utils.h"
#include "cpuThreads.h"

using namespace std;


int main (int argc, char* argv[])
{    
    int rank, size, provided;    
    mpi_init(argc, argv, MPI_THREAD_FUNNELED, provided, rank, size);     
    MPI_Barrier( MPI_COMM_WORLD );

    double t1 = MPI_Wtime();
    testThreads(rank); 
    double t2 = MPI_Wtime();  
    double t = t2-t1;
    printf("Rank %d: Time of testThreads: %lf sec\n",rank, t);

    MPI_Finalize();
    return 0;
}