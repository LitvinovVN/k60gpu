#include <mpi.h>
#include <iostream>
#include "sum.h"

using namespace std;

int main (int argc, char* argv[])
{    
    int rank, size, provided;    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    double t1 = MPI_Wtime();
    //testSum2Arrays(rank, size, 1, 62 * 10000000, 0, 20000000);
    //testSum2Arrays(rank, size, 2, 31 * 10000000, 0, 20000000);
    //testSum2Arrays(rank, size, 4, 31 * 5000000, 0, 20000000);
    //testSum2Arrays(rank, size, 8, 31 * 2500000, 0, 20000000);
    //testSum2Arrays(rank, size, 16, 31 * 1250000, 0, 20000000);
    //testSum2Arrays(rank, size, 31, 2 * 10000000, 0, 20000000);
    //testSum2Arrays(rank, size, 32, 31 * 625000, 0, 20000000);
    //testSum2Arrays(rank, size, 62, 1 * 10000000, 0, 20000000);
    //testSum2Arrays(rank, size, 64, 31 * 312500, 0, 20000000);

    testSum2Arrays(rank, size, 32, 2, 4, 3);
    double t2 = MPI_Wtime();  
    double t = t2-t1;
    printf("Rank %d: Time of testThreads: %lf sec\n",rank, t);
    
    MPI_Finalize();
    return 0;
}