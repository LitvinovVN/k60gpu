#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <sstream> // std::ostringstream

#include "gpu.h"


extern "C"
void printGpuParameters(std::string prefixDescr) {	
	int deviceCount;
	cudaDeviceProp devProp;

	cudaGetDeviceCount(&deviceCount);

	
	std::ostringstream ss;
	ss << "------- Printing CUDA-compatible device properties -------" << std::endl;
    ss << prefixDescr << std::endl;
	ss << "Found " << deviceCount << " cuda compatible devices" << std::endl;

	for(int device = 0; device < deviceCount; device++){
		cudaGetDeviceProperties(&devProp, device);

		ss << "  --- Device " << device << " ---" << std::endl;
		ss << "Compute capability              : " << devProp.major << "." << devProp.minor << std::endl;
		ss << "Name                            : " << devProp.name << std::endl;
		ss << "Total Global Memory             : " << devProp.totalGlobalMem << " bytes ("<< devProp.totalGlobalMem  / 1024.0 / 1024.0 / 1024.0 << " Gb)" << std::endl;
		ss << "Shared memory per block         : " << devProp.sharedMemPerBlock << " bytes" << std::endl;
		ss << "Shared memory per multiprocessor: " << devProp.sharedMemPerMultiprocessor << " bytes" << std::endl;
		ss << "Registers per block             : " << devProp.regsPerBlock << std::endl;
		ss << "Registers per multiprocessor    : " << devProp.regsPerMultiprocessor << std::endl;
		ss << "Warp size                       : " << devProp.warpSize << std::endl;
		ss << "Max threads per block           : " << devProp.maxThreadsPerBlock << std::endl;
		ss << "Total constant memory           : " << devProp.totalConstMem << " bytes" << std::endl;
		ss << "Clock rate                      : " << devProp.clockRate << " Hz" << std::endl;
		ss << "Texture Alignment               : " << devProp.textureAlignment << std::endl;
		ss << "Device Overlap                  : " << devProp.deviceOverlap << std::endl;
		ss << "Multiprocessor Count            : " << devProp.multiProcessorCount << std::endl;
		ss << "Max Threads Dim                 : " << devProp.maxThreadsDim[0] << " " << devProp.maxThreadsDim[1] << " " << devProp.maxThreadsDim[2] << std::endl;
		ss << "Max Threads per block           : " << devProp.maxThreadsPerBlock[0] << " " << devProp.maxThreadsPerBlock[1] << " " << devProp.maxThreadsPerBlock[2] << std::endl;
		ss << "Max Threads per multiprocessor  : " << devProp.maxThreadsPerMultiProcessor[0] << " " << devProp.maxThreadsPerMultiProcessor[1] << " " << devProp.maxThreadsPerMultiProcessor[2] << std::endl;
		ss << "Max Grid Num                    : " << devProp.maxGridSize[0] << " " << devProp.maxGridSize[1] << " " << devProp.maxGridSize[2] << std::endl;
	}

    std::cout << ss.str();
}


__global__ void mult(int x, int y, int *res) {	
	*res = x * y;	
}

extern "C"
int gpu(int x, int y){
	int *dev_res;	
	int res = 0;	
	cudaMalloc((void**)&dev_res, sizeof(int));	
	mult<<<1,1>>>(x, y, dev_res);	
	cudaMemcpy(&res, dev_res, sizeof(int), cudaMemcpyDeviceToHost);	
	cudaFree(dev_res);
	
	return res;
}