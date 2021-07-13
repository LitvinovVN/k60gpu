#include <mpi.h>
#include <iostream>

#include <thread>
#include <vector>

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
    cout << "--- nGpu = "                 << numGpu               << std::endl;
    cout << "--- numElementsPerGpu = "    << numElementsPerGpu    << std::endl;
    cout << "----------------------------------------" << std::endl;

    size_t numElementsInNode = cpuThreadsPerNode * numElementsPerThread + numGpu * numElementsPerGpu;
    size_t numElements = mpi_size * numElementsInNode;

    cout << "numElementsInNode = " << numElementsInNode    << std::endl;
    cout << "numElements = "       << numElements    << std::endl;

    double* a = (double*)malloc(numElements * sizeof(*a));
    double* b = (double*)malloc(numElements * sizeof(*b));
    double* c = (double*)malloc(numElements * sizeof(*c));
    double* c_par = (double*)malloc(numElements * sizeof(*c_par));

    for(int i = 0; i < numElements; i++)
    {
        a[i] = i;
        b[i] = 2.0 * i;
    }

    // Последовательное суммирование
    double t1 = MPI_Wtime();
    for(int i = 0; i < numElements; i++)
    {
        c[i] = a[i] + b[i];        
    }
    double t2 = MPI_Wtime();  
    double t = t2-t1;
    printf("Time of sequential summation: %lf sec\n", t);
    
    // Параллельное суммирование
    t1 = MPI_Wtime();
    //sum2Arrays(a, b, c_par, cpuThreadsPerNode, numElementsPerThread);
    std::vector<std::thread> threads;
	for(int i = 0; i < cpuThreadsPerNode; i++) {
		size_t nStart = i * numElementsPerThread;		
		std::thread thr(thread_sum, a, b, c_par, nStart, numElementsPerThread);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}    


    t2 = MPI_Wtime();
    t = t2-t1;
    printf("Time of parallel summation: %lf sec\n", t);

    // Вывод первых 100 элементов массивов
    for(int i = 0; i < 100; i++)
    {
        std::cerr << "a[" << i << "] = " << a[i] << "; b[" << i << "] = " << b[i] << "; c[" << i << "] = " << c[i] << "; c_par[" << i << "] = " << c_par[i] << std::endl;
    }

    // Вывод неверно рассчитанных элементов массивов
    for(int i = 0; i < numElements; i++)
    {
        if(c[i]-c_par[i] > 0.001)
        {
            std::cerr << "ERROR! Checking stopped! " << "a[" << i << "] = " << a[i] << "; b[" << i << "] = " << b[i] << "; c[" << i << "] = " << c[i] << "; c_par[" << i << "] = " << c_par[i] << std::endl;
            break;
        }
    }

}

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
    testSum2Arrays(rank, size, 16, 31 * 1250000, 0, 20000000);
    //testSum2Arrays(rank, size, 31, 2 * 10000000, 0, 20000000);
    //testSum2Arrays(rank, size, 32, 31 * 625000, 0, 20000000);
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