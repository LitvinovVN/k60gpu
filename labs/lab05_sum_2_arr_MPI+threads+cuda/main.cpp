#include <mpi.h>
#include <iostream>

#include "utils.h"
//#include "cpuThreads.h"
//#include "gpu.h"

#include "sum.h"

using namespace std;


int main (int argc, char* argv[])
{    
    int rank, size, provided;    
    mpi_init(argc, argv, MPI_THREAD_FUNNELED, provided, rank, size);     
    MPI_Barrier( MPI_COMM_WORLD );

    double t1 = MPI_Wtime();
    //testSum2Arrays(rank, size, 1, 62 * 10000000, 0, 20000000);
    //testSum2Arrays(rank, size, 2, 31 * 10000000, 0, 20000000);
    //testSum2Arrays(rank, size, 4, 31 * 5000000, 0, 20000000);
    //testSum2Arrays(rank, size, 8, 31 * 2500000, 0, 20000000);
    //testSum2Arrays(rank, size, 16, 31 * 1250000, 0, 20000000);
    //testSum2Arrays(rank, size, 31, 2 * 10000000, 0, 20000000);
    testSum2Arrays(rank, size, 32, 31 * 625000, 0, 20000000);
    //testSum2Arrays(rank, size, 62, 1 * 10000000, 0, 20000000);
    //testSum2Arrays(rank, size, 64, 31 * 312500, 0, 20000000);

    //testSum2Arrays(rank, size, 62, 10000000, 4, 20000000);
    double t2 = MPI_Wtime();  
    double t = t2-t1;
    printf("Rank %d: Time of testThreads: %lf sec\n",rank, t);

    //printGpuParameters("Node " + std::to_string(rank));

    //int res_gpu = gpu(5, 15);
    //std::cerr<<"res_gpu = "<<res_gpu<<" (rank = "<<rank<<")"<<std::endl;

    //multiGpuTest();

    //multiGpuTest2();

    MPI_Finalize();
    return 0;
}