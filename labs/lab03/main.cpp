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
    //g_lock.lock();

    auto start = std::chrono::system_clock::now();
    int pauseTime = 2; //rand()%4;
    std::this_thread::sleep_for(std::chrono::seconds(pauseTime));
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
   
    fprintf(stderr, "Hostname: %s. MPI rank: %d. Process ID: %d. Thread index: %d. pauseTime = %d. finished computation at %jd. elapsed time: %d \n",
        hostname, rank, getpid(), tnum, pauseTime, (intmax_t)std::ctime(&end_time), elapsed_seconds.count());
    //g_lock.unlock();
}

int main (int argc, char* argv[])
{
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);// MPI_THREAD_MULTIPLE
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
       

    if(rank==0){
        cout << "MPI size is " << size << endl;

        if(provided < MPI_THREAD_FUNNELED)// MPI_THREAD_MULTIPLE
        {
            printf("The threading support level is lesser than that demanded.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        else
        {
            printf("The threading support level corresponds to that demanded.\n");
        }

        auto start = std::chrono::system_clock::now();
        // Some computation here - 1 sec pause for test
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = end-start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);

        std::cout << "finished computation at " << std::ctime(&end_time)
                << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }

    double t = MPI_Wtime();
    char hostname[50];    
    gethostname(hostname, 50);
    pid_t pid = getpid();
    int nHardwareThreads = 8;// std::thread::hardware_concurrency();
    fprintf(stderr, "Time: %lf. Hostname: %s. MPI rank: %d. Process ID: %d. Hardware threads: %d \n", t, hostname, rank, pid, nHardwareThreads);

    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    std::vector<std::thread> threads;
	for(int i = 0; i < nHardwareThreads-1; i++) {
		std::thread thr(thread_proc, i, hostname, rank);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}
    double t2 = MPI_Wtime();
    printf("That took %f seconds\n",t2-t1);


    //---------------------
    std::thread thread1([]{
        fprintf(stderr, "Thread thread1 started... \n" );
        std::this_thread::sleep_for(std::chrono::seconds(2));
        fprintf(stderr, "Thread thread1 woke up after 2 \n" ); });

    std::thread thread2([]{
        fprintf(stderr, "Thread thread2 started... \n" );
        std::this_thread::sleep_for(std::chrono::seconds(1));
        fprintf(stderr, "Thread thread2 woke up after 1 \n" ); });

    std::thread thread3([]{
        fprintf(stderr, "Thread thread3 started... \n" );
        std::this_thread::sleep_for(std::chrono::seconds(3));
        fprintf(stderr, "Thread thread3 woke up after 3 \n" ); });
    
    thread1.join();
    thread2.join();
    thread3.join();

    double t3 = MPI_Wtime();
    printf("Working of thread1, thread2, thread3: %f seconds. Expected 3 seconds\n",t3-t2);
    //---------------------


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