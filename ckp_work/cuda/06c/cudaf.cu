#define CUDA_C_PREF extern "C"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "cudafs.h"
CUDA_C_PREF double MyTime()
{
 struct timeval tv;
 struct timezone tz;
 gettimeofday(&tv, &tz);
 return (double)(tv.tv_sec) + (double)(tv.tv_usec)*1e-6;
}
CUDA_C_PREF int MyCudaDevCount()
{
 int deviceCount;
 cudaGetDeviceCount ( &deviceCount );
 cudaGetDeviceCount ( &deviceCount );
 return deviceCount;
}
CUDA_C_PREF int MyCudaSetDev(int device) 
{
 cudaError_t get;
 get = cudaSetDevice(device);
 if (get != cudaSuccess)
 return -1;
 else
 return 0;
}
CUDA_C_PREF int MyCudaProcess(int mp, int mt,
 int ngb, int ngt,
 int i1, int i2,
 double a, double h,
 double *s)
{
 cudaError_t get;
 double *S_CPU;
 double *S_GPU;
 double t;
 int ndev, mdev, ldev;
 cudaGetDeviceCount (&ndev);
 cudaGetDeviceCount (&ndev);
 if (ndev < 1) return -1;
 if (ndev > 4) return -2;
 fprintf(stderr,"[%02d,%02d]: ndev=%d\n",mp,mt,ndev);
 mdev = mp % ndev;
 get = cudaSetDevice(mdev);
 if (get != cudaSuccess) return 1;
 get = cudaGetDevice(&ldev);
 if (get != cudaSuccess) return 2;
 fprintf(stderr,"[%02d,%02d]: mdev=%d %d\n",mp,mt,mdev,ldev);
 t = MyTime();
 get = cudaMallocHost ((void **)&S_CPU, ngb * sizeof(double));
 if (get != cudaSuccess) return 3;
 get = cudaMalloc ((void **)&S_GPU, ngb * sizeof(double));
 if (get != cudaSuccess) return 4;
 {
 dim3 threads (ngt, 1);
 dim3 blocks (ngb, 1);
 MyCudaProcessReal<<<blocks, threads>>>(i1,i2,a,h,S_GPU);
 }
 get = cudaMemcpy (S_CPU, S_GPU, ngb * sizeof(double), cudaMemcpyDeviceToHost);
 if (get != cudaSuccess) return 5;
 {
 int i;
 double p = 0;
 for (i=0; i<ngb; i++) p += S_CPU[i];
 *s = p;
 }
 get = cudaFree(S_GPU);
 if (get != cudaSuccess) return 6;
 get = cudaFreeHost(S_CPU);
 if (get != cudaSuccess) return 7; 
 t = MyTime() - t;
 fprintf(stderr,"[%02d,%02d]: time=%.6lf\n",mp,mt,t);
 return 0;
}
__device__ void MyRange(int np, int mp, int ia, int ib,
 int *i1, int *i2, int *nc)
{
 if (np<2) { *i1=ia; *i2=ib; *nc=ib-ia+1; }
 else {
 int ni, mi, nn;
 nn = ib - ia + 1; ni = nn / np; mi = nn - ni * np;
 if (mp+1<=mi)
 { *i1 = ia + mp * (ni+1); *i2 = *i1 + ni; }
 else
 { *i1 = ia + mi * (ni+1) + (mp-mi) * ni; *i2 = *i1 + ni - 1; }
 *nc = *i2 - *i1 + 1;
 }
 return;
}
__device__ double MyFun(double x)
{
 double c = cos(x);
 c = c * tan(x-0.25);
 c = c * tan(x-0.5);
 c = c * exp(-x);
 c = c * log(1.25+x);
 return 4.0 / (1.0 + x * x * (2.0 - c) / (2.0 + c));
}
__global__ void MyCudaProcessReal(int i1, int i2,
 double a, double h,
 double *DATAOUT)
{
 __shared__ double cache[1024]; // 1 <= thread count <= 512
 int cacheIndex = threadIdx.x;
 int ng = blockDim.x * gridDim.x;
 int mg = blockDim.x * blockIdx.x + threadIdx.x;
 int j, j1, j2, jc;
 int k, kk = 20;
 double s=0;
 MyRange(ng,mg,i1,i2,&j1,&j2,&jc);
 for (k=0; k<kk; k++)
 for (j=j1; j<=j2; j++) {
 double x = a + h * (1.0 * j - 0.5);
 s += MyFun(x) * h;
 }
 s /= (1.0*kk);
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