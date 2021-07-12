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
		ss << "Clock rate                      : " << devProp.clockRate << " kHz" << std::endl;
		ss << "Global memory bus width         : " << devProp.memoryBusWidth << " bits" << std::endl;
		ss << "Peak memory clock frequency     : " << devProp.memoryClockRate << " kHz" << std::endl;
		ss << "Texture alignment               : " << devProp.textureAlignment << std::endl;
		ss << "Device overlap                  : " << devProp.deviceOverlap << std::endl;
		ss << "Multiprocessor count            : " << devProp.multiProcessorCount << std::endl;
		ss << "Max threads dim                 : " << devProp.maxThreadsDim[0] << " " << devProp.maxThreadsDim[1] << " " << devProp.maxThreadsDim[2] << std::endl;
		ss << "Max threads per block           : " << devProp.maxThreadsPerBlock << std::endl;
		ss << "Max threads per multiprocessor  : " << devProp.maxThreadsPerMultiProcessor << std::endl;
		ss << "Max grid num                    : " << devProp.maxGridSize[0] << " " << devProp.maxGridSize[1] << " " << devProp.maxGridSize[2] << std::endl;
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


//////////////////////////////////////////////////////////////////////
#define imin(a,b) (a<b?a:b)

#define     N    (33*1024*1024)
	const int threadsPerBlock = 256;
	const int blocksPerGrid =
            imin( 32, (N/2+threadsPerBlock-1) / threadsPerBlock );

static void HandleError( cudaError_t err,
	const char *file,
	int line ) {
		if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
		file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
	   printf( "Host memory failed in %s at line %d\n", \
			   __FILE__, __LINE__ ); \
	   exit( EXIT_FAILURE );}}



#if _WIN32
   //Windows threads.
   #include <windows.h>
   
   typedef HANDLE CUTThread;
   typedef unsigned (WINAPI *CUT_THREADROUTINE)(void *);
   
   #define CUT_THREADPROC unsigned WINAPI
   #define  CUT_THREADEND return 0
   
#else
   //POSIX threads.
   #include <pthread.h>
   
   typedef pthread_t CUTThread;
   typedef void *(*CUT_THREADROUTINE)(void *);
   
   #define CUT_THREADPROC void
   #define  CUT_THREADEND
  #endif
   
//Create thread.
CUTThread start_thread( CUT_THREADROUTINE, void *data );
   
//Wait for thread to finish.
void end_thread( CUTThread thread );
   
//Destroy thread.
void destroy_thread( CUTThread thread );
   
//Wait for multiple threads.
void wait_for_threads( const CUTThread *threads, int num );
   
#if _WIN32
   //Create thread
   CUTThread start_thread(CUT_THREADROUTINE func, void *data){
	   return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
   }
   
   //Wait for thread to finish
   void end_thread(CUTThread thread){
	   WaitForSingleObject(thread, INFINITE);
	   CloseHandle(thread);
   }
   
   //Destroy thread
   void destroy_thread( CUTThread thread ){
	   TerminateThread(thread, 0);
	   CloseHandle(thread);
   }
   
   //Wait for multiple threads
   void wait_for_threads(const CUTThread * threads, int num){
	   WaitForMultipleObjects(num, threads, true, INFINITE);
   
	   for(int i = 0; i < num; i++)
	   CloseHandle(threads[i]);
   }
   
#else
   //Create thread
   CUTThread start_thread(CUT_THREADROUTINE func, void * data){
	   pthread_t thread;
	   pthread_create(&thread, NULL, func, data);
	   return thread;
   }
   
   //Wait for thread to finish
   void end_thread(CUTThread thread){
	   pthread_join(thread, NULL);
   }
   
   //Destroy thread
   void destroy_thread( CUTThread thread ){
	   pthread_cancel(thread);
   }
   
   //Wait for multiple threads
   void wait_for_threads(const CUTThread * threads, int num){
	   for(int i = 0; i < num; i++)
		   end_thread( threads[i] );
	}
   
#endif



__global__ void dot( int size, float *a, float *b, float *c ) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < size) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


struct DataStruct {
    int     deviceID;
    int     size;
    float   *a;
    float   *b;
    float   returnValue;
};


void* routine( void *pvoidData ) {
    DataStruct  *data = (DataStruct*)pvoidData;
    HANDLE_ERROR( cudaSetDevice( data->deviceID ) );

    int     size = data->size;
    float   *a, *b, c, *partial_c;
    float   *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the CPU side
    a = data->a;
    b = data->b;
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a,
                              size*sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b,
                              size*sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c,
                              blocksPerGrid*sizeof(float) ) );

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, size*sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, size*sizeof(float),
                              cudaMemcpyHostToDevice ) ); 

    dot<<<blocksPerGrid,threadsPerBlock>>>( size, dev_a, dev_b,
                                            dev_partial_c );
    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( partial_c, dev_partial_c,
                              blocksPerGrid*sizeof(float),
                              cudaMemcpyDeviceToHost ) );

    // finish up on the CPU side
    c = 0;
    for (int i=0; i<blocksPerGrid; i++) {
        c += partial_c[i];
    }

    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_partial_c ) );

    // free memory on the CPU side
    free( partial_c );

    data->returnValue = c;
    return 0;
}




extern "C" void multiGpuTest(){
	std::cerr << "multiGpuTest()" << std::endl;

	int deviceCount;
    HANDLE_ERROR( cudaGetDeviceCount( &deviceCount ) );
    if (deviceCount < 2) {
        printf( "We need at least two compute 1.0 or greater "
                "devices, but only found %d\n", deviceCount );
        return;
    }

    float   *a = (float*)malloc( sizeof(float) * N );
    HANDLE_NULL( a );
    float   *b = (float*)malloc( sizeof(float) * N );
    HANDLE_NULL( b );

    // fill in the host memory with data
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    // prepare for multithread
    DataStruct  data[2];
    data[0].deviceID = 0;
    data[0].size = N/2;
    data[0].a = a;
    data[0].b = b;

    data[1].deviceID = 1;
    data[1].size = N/2;
    data[1].a = a + N/2;
    data[1].b = b + N/2;

    CUTThread   thread = start_thread( routine, &(data[0]) );
    routine( &(data[1]) );
    end_thread( thread );


    // free memory on the CPU side
    free( a );
    free( b );

    printf( "Value calculated:  %f\n",
            data[0].returnValue + data[1].returnValue );
}

////////////////////////////////////////////////////////////////////////////
//// https://coderoad.ru/57173023/использование-std-thread-и-CUDA-вместе ///
#include <thread>
#include <vector>
//#include <iostream>
//#include <cstdio>

__global__ void printHelloFromThreadN_kernel(int n){
	printf("hello from thread %d\n", n);	
}
  
void thread_func(int n){
	cudaSetDevice(n);
	printHelloFromThreadN_kernel<<<1,1>>>(n);
	cudaDeviceSynchronize();
}


extern "C" void multiGpuTest2(){
	std::cerr << "multiGpuTest2()" << std::endl;

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
}