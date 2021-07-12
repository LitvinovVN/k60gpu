#include <mpi.h>
#include <iostream> 

#include "utils.h"
#include "cpuThreads.h"
#include "gpu.h"

using namespace std;

void testSum2Arrays(int mpi_rank, int mpi_size,
                    int cpuThreadsPerNode, int numElementsPerThread,
                    int numGpu, int numElementsPerGpu)
{
    cout << "----------------------------------------" << std::endl;
    cout << "-------------testSum2Arrays-------------" << std::endl;
    cout << "--- mpi_rank = "             << mpi_rank             << std::endl;
    cout << "--- mpi_size = "             << mpi_size             << std::endl;
    cout << "--- cpuThreadsPerNode = "    << cpuThreadsPerNode    << std::endl;
    cout << "--- numElementsPerThread = " << numElementsPerThread << std::endl;
    cout << "--- nGpu = "                 << nGpu                 << std::endl;
    cout << "--- numElementsPerGpu = "    << numElementsPerGpu    << std::endl;
    cout << "----------------------------------------" << std::endl;

    size_t numElementsInNode = cpuThreadsPerNode * numElementsPerThread + nGpu * numElementsPerGpu
    size_t numElements = mpi_size * numElementsInNode;
}

int main (int argc, char* argv[])
{    
    int rank, size, provided;    
    mpi_init(argc, argv, MPI_THREAD_FUNNELED, provided, rank, size);     
    MPI_Barrier( MPI_COMM_WORLD );

    double t1 = MPI_Wtime();
    testSum2Arrays(rank, size, 62, 1000, 4, 10000);
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