
#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>
#include "sum.h"



__global__ void sum_kernel(double* a, double* b, double* c_par, int dev_indx, int nStart, int nBlocks, int nThreads, int numElementsPerGpuThread){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //if(tid == 0)
    //{
    //    printf("dev_indx = %d | nStart = %d | nBlocks = %d | nThreads = %d | numElementsPerGpuThread = %d\n", dev_indx, nStart, nBlocks, nThreads, numElementsPerGpuThread);
    //}  

    size_t indx_start = nStart + tid * numElementsPerGpuThread;
    for(int i = indx_start; i < indx_start + numElementsPerGpuThread; i++)
    {        
        c_par[i] = a[i] + b[i];
    }
}
  
void thread_sum_gpu(int dev_indx, double* a, double* b, double* c_par, size_t nStart, size_t nBlocks, size_t nThreads, size_t numElementsPerGpuThread){
	cudaSetDevice(dev_indx);
    dim3 blocks = dim3(nBlocks);
    dim3 threads = dim3(nThreads);
	sum_kernel<<<blocks, threads>>>(a, b, c_par, dev_indx, nStart, nBlocks, nThreads, numElementsPerGpuThread);
	cudaDeviceSynchronize();
}


void thread_sum(double* a, double* b, double* c_par, size_t nStart, size_t numElementsPerThread) {
    for(int indx = nStart; indx < nStart+numElementsPerThread; indx++)
	{
		c_par[indx] = a[indx] + b[indx];
	}
}

void sum2Arrays(double* a, double* b, double* c_par, size_t cpuThreadsPerNode, size_t numElementsPerThread,
    size_t numGpu, size_t nBlocks, size_t nThreads, size_t numElementsPerGpuThread){
    // CPU threads starting
    std::vector<std::thread> t_cpu_vec;
	for(int i = 0; i < cpuThreadsPerNode; i++) {
		size_t nStart = i * numElementsPerThread;
		std::thread thr(thread_sum, a, b, c_par, nStart, numElementsPerThread);
		t_cpu_vec.emplace_back(std::move(thr));
	}
	
    // GPU threads starting
  	std::vector<std::thread> t_gpu_vec;
    size_t nGpuStart = cpuThreadsPerNode * numElementsPerThread;    
    size_t numElementsPerGpu = nBlocks * nThreads * numElementsPerGpuThread;
  	for (int i = 0; i < numGpu; i++){
        size_t nStart = nGpuStart + i * numElementsPerGpu;
        t_gpu_vec.push_back(std::thread(thread_sum_gpu, i, a, b, c_par, nStart, nBlocks, nThreads, numElementsPerGpuThread)); 
    } 	

    // GPU threads waiting
  	for (int i = 0; i < numGpu; i++)
        t_gpu_vec[i].join();  	
    
    // CPU threads waiting
	for(auto& t_cpu : t_cpu_vec) {
		t_cpu.join();
	}    
}



extern "C"
void testSum2Arrays(int mpi_rank, int mpi_size,
    size_t cpuThreadsPerNode, size_t numElementsPerThread,
    size_t numGpu, size_t nBlocks, size_t nThreads, size_t numElementsPerGpuThread)
{
    std::cout << "----------------------------------------"                     << std::endl;
    std::cout << "-------------testSum2Arrays-------------"                     << std::endl;
    std::cout << "--- mpi_rank = "                  << mpi_rank                 << std::endl;
    std::cout << "--- mpi_size = "                  << mpi_size                 << std::endl;
    std::cout << "--- cpuThreadsPerNode = "         << cpuThreadsPerNode        << std::endl;
    std::cout << "--- numElementsPerThread = "      << numElementsPerThread     << std::endl;
    std::cout << "--- nGpu = "                      << numGpu                   << std::endl;
    std::cout << "--- nBlocks = "                   << nBlocks                  << std::endl;
    std::cout << "--- nThreads = "                  << nThreads                 << std::endl;
    std::cout << "--- numElementsPerGpuThread = "   << numElementsPerGpuThread  << std::endl;
    std::cout << "----------------------------------------"                     << std::endl;

    size_t numElementsInNodeCpu = cpuThreadsPerNode * numElementsPerThread;
    size_t numElementsInNodeGpu = nBlocks * nThreads * numElementsPerGpuThread;
    size_t numElementsInNode = numElementsInNodeCpu + numElementsInNodeGpu;
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
    // ???????????????????????????????? ????????????????????????
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

    // ???????????????????????? ????????????????????????
    cudaEvent_t startPar, stopPar;
    cudaEventCreate(&startPar);
    cudaEventCreate(&stopPar);
    cudaEventRecord(startPar, 0);
    auto start_sc = std::chrono::system_clock::now();
    sum2Arrays(a, b, c_par, cpuThreadsPerNode, numElementsPerThread,
        numGpu, nBlocks, nThreads, numElementsPerGpuThread);
    auto end_sc = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_sc-start_sc;
    printf("Time of parallel summation system_clock: %lf sec\n", elapsed_seconds);
    cudaEventRecord(stopPar, 0); 
    cudaEventSynchronize(stopPar);    
    float elapsedTimePar;
    cudaEventElapsedTime(&elapsedTimePar, startPar, stopPar);
    printf("Time of parallel summation: %lf sec\n", elapsedTimePar/1000);

    // ?????????? ?????????????????? ????????????????
    /*for(int i = 0; i < numElements; i++)
    {
        std::cerr << "a[" << i << "] = " << a[i] << "; b[" << i << "] = " << b[i] << "; c[" << i << "] = " << c[i] << "; c_par[" << i << "] = " << c_par[i] << std::endl;
    }*/
    int i = 0;
    std::cerr << "a[" << i << "] = " << a[i] << "; b[" << i << "] = " << b[i] << "; c[" << i << "] = " << c[i] << "; c_par[" << i << "] = " << c_par[i] << std::endl;
    i = numElements - 1;
    std::cerr << "a[" << i << "] = " << a[i] << "; b[" << i << "] = " << b[i] << "; c[" << i << "] = " << c[i] << "; c_par[" << i << "] = " << c_par[i] << std::endl;

    // ?????????? ?????????????? ???????????????????????? ?????????????????? ????????????????
    bool isCorrect = true;
    for(int i = 0; i < numElements; i++)
    {
        if(c[i]-c_par[i] > 0.001)
        {
            isCorrect = false;
            std::cerr << "ERROR! Checking stopped! " << "a[" << i << "] = " << a[i] << "; b[" << i << "] = " << b[i] << "; c[" << i << "] = " << c[i] << "; c_par[" << i << "] = " << c_par[i] << std::endl;
            break;
        }
    }

    if(isCorrect)
        std::cerr << "Checking successed!" << std::endl;    

}
