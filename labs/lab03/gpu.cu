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

	//std::cerr << prefixDescr << " printGpuParameters(): " << "deviceCount = " << deviceCount << std::endl;

	std::ostringstream ss;
    ss << prefixDescr << std::endl;
	ss << "Finded " << deviceCount << " cuda compatible devices" << std::endl;

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