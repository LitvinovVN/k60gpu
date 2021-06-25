#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <thread>
#include <vector>

#include "cpu.h"
#include "gpu.h"

using namespace std;

void thread_proc(int tnum, char* hostname, int rank) {
    fprintf(stderr, "Time: %lf. Hostname: %s. MPI rank: %d. Process ID: %d. Thread index: %d \n", MPI_Wtime(), hostname, rank, getpid(), tnum);
}

int main (int argc, char* argv[])
{
    int rank, size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    if(rank==0)
        cout << "MPI size is " << size << endl;

    double t = MPI_Wtime();
    char hostname[50];    
    gethostname(hostname, 50);
    pid_t pid = getpid();
    int nHardwareThreads = std::thread::hardware_concurrency();
    fprintf(stderr, "Time: %lf. Hostname: %s. MPI rank: %d. Process ID: %d. Hardware threads: %d \n", t, hostname, rank, pid, nHardwareThreads);

    std::vector<std::thread> threads;
	for(int i = 0; i < nHardwareThreads-1; i++) {
		std::thread thr(thread_proc, i, hostname, rank);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}

    MPI_Barrier(MPI_COMM_WORLD);

    int res_gpu = gpu(x, y);	
	int res_cpu = cpu(x, y);
	
	std::cout<<"res_gpu = "<<res_gpu<<" (rank = "<<rank<<")"<<std::endl;
	std::cout<<"res_cpu = "<<res_cpu<<" (rank = "<<rank<<")"<<std::endl;

	MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}