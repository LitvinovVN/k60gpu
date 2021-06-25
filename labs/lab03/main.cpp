#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <thread>
#include <vector>

#include "cpu.h"
#include "gpu.h"

using namespace std;

void thread_proc(int tnum, char* hostname, int rank) {
    double start = MPI_Wtime();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    fprintf(stderr, "Time: %lf s. Hostname: %s. MPI rank: %d. Process ID: %d. Thread index: %d \n", MPI_Wtime() - start, hostname, rank, getpid(), tnum);
}

int main (int argc, char* argv[])
{
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
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

    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    std::vector<std::thread> threads;
	for(int i = 0; i < 2 * nHardwareThreads-1; i++) {
		std::thread thr(thread_proc, i, hostname, rank);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}
    double t2 = MPI_Wtime();
    printf("That took %f seconds\n",t2-t1);

    MPI_Barrier(MPI_COMM_WORLD);

    int x = rank;
    int y = 100;
    int res_gpu = gpu(x, y);	
	int res_cpu = cpu(x, y);
	
    fprintf(stderr, "Time: %lf. Hostname: %s. MPI rank: %d. Process ID: %d. res_gpu = %d.  res_cpu = %d\n", t, hostname, rank, pid, res_gpu, res_cpu);
	
	MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}