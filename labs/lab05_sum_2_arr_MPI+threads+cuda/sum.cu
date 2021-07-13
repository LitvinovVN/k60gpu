
#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>
#include "sum.h"


void thread_sum(double* a, double* b, double* c_par, size_t nStart, size_t numElementsPerThread) {
    for(int indx = nStart; indx < nStart+numElementsPerThread; indx++)
	{
		c_par[indx] = a[indx] + b[indx];
	}
	
	//fprintf(stderr, "Thread started (thread_sum)... %d %d \n", nStart, nStart+numElementsPerThread);  

    //fprintf(stderr, "Time: %lf. MPI rank: %d. Process ID: %d. Thread index: %d. pauseTime = %d ms \n", MPI_Wtime(), rank, getpid(), tnum, pauseTime);    
}

void sum2Arrays(double* a, double* b, double* c_par, size_t cpuThreadsPerNode, size_t numElementsPerThread){
    std::vector<std::thread> threads;
	for(int i = 0; i < cpuThreadsPerNode; i++) {
		size_t nStart = i * numElementsPerThread;		
		std::thread thr(thread_sum, a, b, c_par, nStart, numElementsPerThread);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}    
}



void multiGpuSum2Arrays(){
    std::cout << "multiGpuSum2Arrays()" << std::endl;
}


extern "C"
void testSum2Arrays(int mpi_rank, int mpi_size,
    int cpuThreadsPerNode, int numElementsPerThread,
    int numGpu, int numElementsPerGpu)
{
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "-------------testSum2Arrays-------------" << std::endl;
    std::cout << "--- mpi_rank = "             << mpi_rank             << std::endl;
    std::cout << "--- mpi_size = "             << mpi_size             << std::endl;
    std::cout << "--- cpuThreadsPerNode = "    << cpuThreadsPerNode    << std::endl;
    std::cout << "--- numElementsPerThread = " << numElementsPerThread << std::endl;
    std::cout << "--- nGpu = "                 << numGpu               << std::endl;
    std::cout << "--- numElementsPerGpu = "    << numElementsPerGpu    << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    size_t numElementsInNode = cpuThreadsPerNode * numElementsPerThread + numGpu * numElementsPerGpu;
    size_t numElements = mpi_size * numElementsInNode;

    std::cout << "numElementsInNode = " << numElementsInNode    << std::endl;
    std::cout << "numElements = "       << numElements    << std::endl;

    //double* a = (double*)malloc(numElements * sizeof(*a));
    //double* b = (double*)malloc(numElements * sizeof(*b));
    //double* c = (double*)malloc(numElements * sizeof(*c));
    //double* c_par = (double*)malloc(numElements * sizeof(*c_par));
    
    double* a;
    double* b;
    double* c;
    double* c_par;
    cudaHostAlloc((void**)&a, numElements * sizeof(*a), cudaHostAllocDefault);
    cudaHostAlloc((void**)&b, numElements * sizeof(*b), cudaHostAllocDefault);
    cudaHostAlloc((void**)&c, numElements * sizeof(*c), cudaHostAllocDefault);
    cudaHostAlloc((void**)&c_par, numElements * sizeof(*c_par), cudaHostAllocDefault);
    

    for(int i = 0; i < numElements; i++)
    {
        a[i] = i;
        b[i] = 2.0 * i;
    }

    // Последовательное суммирование
    //double t1 = MPI_Wtime();
    for(int i = 0; i < numElements; i++)
    {
        c[i] = a[i] + b[i];        
    }
    //double t2 = MPI_Wtime();  
    //double t = t2-t1;
    //printf("Time of sequential summation: %lf sec\n", t);

    // Параллельное суммирование
    //t1 = MPI_Wtime();
    sum2Arrays(a, b, c_par, cpuThreadsPerNode, numElementsPerThread);

    // GPU start
    multiGpuSum2Arrays();    

    //t2 = MPI_Wtime();
    //t = t2-t1;
    //printf("Time of parallel summation: %lf sec\n", t);

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
