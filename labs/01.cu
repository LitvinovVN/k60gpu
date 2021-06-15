#include <stdio.h>

//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//

#define N 1000

using namespace std;

//
// A function marked __global__
// runs on the GPU but can be called from
// the CPU.
//
// This function multiplies the elements of an array
// of ints by 2.
//
// The entire computation can be thought of as running
// with one thread per array element with blockIdx.x
// identifying the thread.
//
// The comparison i<N is because often it isn't convenient
// to have an exact 1-1 correspondence between threads
// and array elements. Not strictly necessary here.
//
// Note how we're mixing GPU and CPU code in the same source
// file. An alternative way to use CUDA is to keep
// C/C++ code separate from CUDA code and dynamically
// compile and load the CUDA code at runtime, a little
// like how you compile and load OpenGL shaders from
// C/C++ code.
//
__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}


void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    wcout << "NBody.GPU" << endl << "=========" << endl << endl;

    wcout << "CUDA version:   v" << CUDART_VERSION << endl;    
    wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl; 

    int devCount;
    cudaGetDeviceCount(&devCount);
    wcout << "CUDA Devices: " << endl << endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
        wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
        wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
        wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

        wcout << "  Warp size:         " << props.warpSize << endl;
        wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
        wcout << endl;
    }
}


int main() {
    DisplayHeader();

    //
    // Create int arrays on the CPU.
    // ('h' stands for "host".)
    //
    int ha[N], hb[N];

    //
    // Create corresponding int arrays on the GPU.
    // ('d' stands for "device".)
    //
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    //
    // Initialise the input data on the CPU.
    //
    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }

    //
    // Copy input data to array on GPU.
    //
    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

    //
    // Launch GPU code with N threads, one per
    // array element.
    //
    add<<<N, 1>>>(da, db);

    

    //
    // Copy output array from GPU back to CPU.
    //
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }

    //
    // Free up the arrays on the GPU.
    //
    cudaFree(da);
    cudaFree(db);

    return 0;
}