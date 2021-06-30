#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <ctime> 

#include "cpu.h"
#include "gpu.h"

using namespace std;

std::mutex g_lock;

void thread_proc(int tnum, char* hostname, int rank) {
    g_lock.lock();

    auto start = std::chrono::system_clock::now();
    int pauseTime = rand()%4;
    std::this_thread::sleep_for(std::chrono::seconds(pauseTime));
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    fprintf(stderr, "Time: %lf s. Hostname: %s. MPI rank: %d. Process ID: %d. Thread index: %d. pauseTime = %d. finished computation at %jd. elapsed time: %d \n",
     MPI_Wtime() - start, hostname, rank, getpid(), tnum, pauseTime, (intmax_t)std::ctime(&end_time), elapsed_seconds.count());
    g_lock.unlock();
}

int main (int argc, char* argv[])
{
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
       

    if(rank==0){
        cout << "MPI size is " << size << endl;

        if(provided < MPI_THREAD_MULTIPLE)
        {
            printf("The threading support level is lesser than that demanded.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        else
        {
            printf("The threading support level corresponds to that demanded.\n");
        }
    }

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