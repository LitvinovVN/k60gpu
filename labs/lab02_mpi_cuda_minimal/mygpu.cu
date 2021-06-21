#include "myplatform.h"

#define CUDA_C_PREF extern "C"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mygpu.h"

#ifdef WIN_PLATFORM
  #include <windows.h>
#else
  #include <unistd.h>
  #include <pthread.h>
  #include <sched.h>
  #include <sys/time.h>
#endif

// Internal functions:

__global__ void MyGPUProcessReal(int i1, int i2,
                                 double a, double h,
                                 double *sum);

__device__ double MyFun(double x);

__device__ void MyRange(int np, int mp, int ia, int ib,
                        int *i1, int *i2, int *nc);

CUDA_C_PREF double mytime( void );

// CPU time:

CUDA_C_PREF double mytime( void )
{
#ifdef WIN_PLATFORM
  return 0;
#else
  struct timeval tv;
  gettimeofday(&tv,0);
  return tv.tv_sec + tv.tv_usec/1000000.;
#endif
}

// GPU count:

CUDA_C_PREF int MyGetGPUCount()
{
  int deviceCount;
  cudaGetDeviceCount ( &deviceCount );		
  return deviceCount;
}

// GPU info:

CUDA_C_PREF int MyGPUInfo(int rank)
{
  cudaDeviceProp devProp;
  int deviceCount;
  int device = 0;

  cudaGetDeviceCount ( &deviceCount );		
  fprintf(stderr,"[%d,%d]: Total number of GPU: %d\n", rank, device, deviceCount);
				
  for (device = 0; device < deviceCount; device++ ) {
    cudaGetDeviceProperties ( &devProp, device );
    fprintf (stderr,"[%d,%d]: Name                    : %s\n", rank, device, devProp.name );
    fprintf (stderr,"[%d,%d]: Compute capability      : %d.%d\n", rank, device, devProp.major, devProp.minor );
    fprintf (stderr,"[%d,%d]: Shared memory per block : %d bytes\n", rank, device, devProp.sharedMemPerBlock );
    fprintf (stderr,"[%d,%d]: Registers per block     : %d x 32 bits = %d bytes\n", rank, device, devProp.regsPerBlock, devProp.regsPerBlock*4);
    fprintf (stderr,"[%d,%d]: Warp size               : %d\n", rank, device, devProp.warpSize );
    fprintf (stderr,"[%d,%d]: Max threads per block   : %d\n", rank, device, devProp.maxThreadsPerBlock );
    fprintf (stderr,"[%d,%d]: Total constant memory   : %d bytes\n", rank, device, devProp.totalConstMem );
    fprintf (stderr,"[%d,%d]: Device overlap          : %d\n", rank, device, devProp.deviceOverlap );
    fprintf (stderr,"[%d,%d]: Multiprocessor count    : %d\n", rank, device, devProp.multiProcessorCount );
    fprintf (stderr,"[%d,%d]: Clock rate              : %d MHz\n", rank, device, devProp.clockRate/1000 );
    fprintf (stderr,"[%d,%d]: Max grid size           : [%8d, %8d, %8d]\n", rank, device, devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    fprintf (stderr,"[%d,%d]: Max threads dim         : [%8d, %8d, %8d]\n", rank, device, devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    fprintf (stderr,"[%d,%d]: Global memory           : %12.6lf MB\n", rank, device, ((double)devProp.totalGlobalMem)/(1024*1024));
    fprintf (stderr,"[%d,%d]: Mem pitch               : %12.6lf MB\n", rank, device, ((double)devProp.memPitch)/(1024*1024));
    fprintf (stderr,"[%d,%d]: Kernel exec timeout     : %d\n", rank, device, devProp.kernelExecTimeoutEnabled );
    fprintf (stderr,"[%d,%d]: ECC enabled             : %d\n", rank, device, devProp.ECCEnabled );
    fprintf (stderr,"[%d,%d]: Cuncurrent kernels      : %d\n", rank, device, devProp.concurrentKernels );
  }

  return deviceCount;
}

// GPU set:

CUDA_C_PREF int MyGPUSetDev(int device)
{
  cudaError_t get;
  get = cudaSetDevice(device);
  if (get != cudaSuccess)
    return -1;
  else
    return 0;
}

// GPU process:

CUDA_C_PREF int MyGPUProcess(int mp, int mt,
                             int ngb, int ngt,
                             int i1, int i2,
                             double a, double h,
                             double *s)
{
  cudaError_t get;
  double *S_CPU;
  double *S_GPU;

  get = cudaSetDevice(mt);
  if (get != cudaSuccess) return 1;

  get = cudaMallocHost ((void **)&S_CPU, ngb * sizeof(double)); 
  if (get != cudaSuccess) return 2;

  get = cudaMalloc ((void **)&S_GPU, ngb * sizeof(double));
  if (get != cudaSuccess) return 3;

  {
    dim3 threads (ngt, 1);
    dim3 blocks  (ngb, 1);
    MyGPUProcessReal<<<blocks, threads>>>(i1,i2,a,h,S_GPU);
  }

  get = cudaMemcpy (S_CPU, S_GPU, ngb * sizeof(double), cudaMemcpyDeviceToHost);		
  if (get != cudaSuccess) return 4;

  {
    int i;
    double p = 0;
    for (i=0; i<ngb; i++) p += S_CPU[i];
    *s = p;
  }

  get = cudaFree(S_GPU);
  if (get != cudaSuccess) return 5;

  get = cudaFreeHost(S_CPU);
  if (get != cudaSuccess) return 6;

  return 0;
}

// MyRange:

__device__ void MyRange(int np, int mp, int ia, int ib,
                        int *i1, int *i2, int *nc)
{
  if (np<2) { *i1=ia; *i2=ib; *nc=ib-ia+1; }
  else {
    int ni, mi, nn;
    nn = ib - ia + 1; ni = nn / np;  mi = nn - ni * np;
    if (mp+1<=mi)
      { *i1 = ia + mp * (ni+1); *i2 = *i1 + ni; }
    else
      { *i1 = ia + mi * (ni+1) + (mp-mi) * ni; *i2 = *i1 + ni - 1; }
    *nc = *i2 - *i1 + 1;
  }
  return;
}

// GPU real process:

__global__ void MyGPUProcessReal(int i1, int i2,
                                 double a, double h,
                                 double *DATAOUT)
{
  __shared__ double cache[1024]; // 1 <= thread count <= 512
  int cacheIndex = threadIdx.x;
  int ng = blockDim.x * gridDim.x;
  int mg = blockDim.x * blockIdx.x + threadIdx.x;
  int j, j1, j2, jc;
  double s=0;

  MyRange(ng,mg,i1,i2,&j1,&j2,&jc);

  for (j=j1; j<=j2; j++) {
    double x = a + h * (1.0 * j - 0.5);
    s += MyFun(x) * h;
  }

// Save to cache & synchronize:
	
  cache[cacheIndex] = s;

  __syncthreads();

// Reduction:

  j = blockDim.x/2;

  while (j != 0) {
    if (cacheIndex < j)
      cache[cacheIndex] += cache[cacheIndex + j];
    __syncthreads();
    j /= 2;
  }

  if (cacheIndex == 0)
    DATAOUT[blockIdx.x] = cache[0];

  __syncthreads();
}

// Job function:

__device__ double MyFun(double x)
{
  double c = cos(x);
  c = c * tan(x-0.25);
  c = c * tan(x-0.5);
  c = c * exp(-x);
  c = c * log(1.25+x);
  return 4.0 / (1.0 + x * x * (2.0 - c) / (2.0 + c));
}
