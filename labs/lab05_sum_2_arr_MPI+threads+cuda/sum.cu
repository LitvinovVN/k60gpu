
#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>
#include "sum.h"



__global__ void printHelloFromThreadN_kernel(int n){
	printf("hello from thread %d\n", n);	
}
  
void thread_func(int n){
	cudaSetDevice(n);
	printHelloFromThreadN_kernel<<<1,1>>>(n);
	cudaDeviceSynchronize();
}




void thread_sum(double* a, double* b, double* c_par, size_t nStart, size_t numElementsPerThread) {
    for(int indx = nStart; indx < nStart+numElementsPerThread; indx++)
	{
		c_par[indx] = a[indx] + b[indx];
	}
}

void sum2Arrays(double* a, double* b, double* c_par, size_t cpuThreadsPerNode, size_t numElementsPerThread,
    size_t numGpu, size_t numElementsPerGpu){
    std::vector<std::thread> threads;
	for(int i = 0; i < cpuThreadsPerNode; i++) {
		size_t nStart = i * numElementsPerThread;		
		std::thread thr(thread_sum, a, b, c_par, nStart, numElementsPerThread);
		threads.emplace_back(std::move(thr));
	}
	
    /////
    int n = 0;
  	cudaError_t err = cudaGetDeviceCount(&n);
  	if (err != cudaSuccess) {std::cout << "error " << (int)err << std::endl; return;}

  	std::vector<std::thread> t;
  	for (int i = 0; i < n; i++)
    	t.push_back(std::thread(thread_func, i));
  	std::cout << n << " threads started" << std::endl;

  	for (int i = 0; i < n; i++)
    	t[i].join();
  	std::cout << "join finished" << std::endl;
    /////

	for(auto& thr : threads) {
		thr.join();
	}    
}



extern "C"
void testSum2Arrays(int mpi_rank, int mpi_size,
    size_t cpuThreadsPerNode, size_t numElementsPerThread,
    size_t numGpu, size_t numElementsPerGpu)
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Последовательное суммирование
    cudaEventRecord(start, 0);
    for(int i = 0; i < numElements; i++)
    {
        c[i] = a[i] + b[i];        
    }
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);
    float elapsedTimeSeq;
    cudaEventElapsedTime(&elapsedTimeSeq, start, stop);
    printf("Time of sequential summation: %lf sec\n", elapsedTimeSeq/1000);

    // Параллельное суммирование
    cudaEventRecord(start, 0);
    sum2Arrays(a, b, c_par, cpuThreadsPerNode, numElementsPerThread,
        numGpu, numElementsPerGpu);
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);    
    float elapsedTimePar;
    cudaEventElapsedTime(&elapsedTimePar, start, stop);
    printf("Time of parallel summation: %lf sec\n", elapsedTimePar/1000);

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
