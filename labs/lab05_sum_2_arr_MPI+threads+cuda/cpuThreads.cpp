#include <mpi.h>
#include <iostream> 
#include <unistd.h> // getpid
#include <thread>
#include <vector>
#include <chrono> // chrono::system_clock
#include <ctime>   // localtime

#include "cpuThreads.h"

extern "C"
void thread_proc(int tnum, int rank) {
    fprintf(stderr, "Thread %d started at node %d... \n", tnum, rank);

    auto start = std::chrono::system_clock::now();
    int pauseTime = tnum * 10;
    std::this_thread::sleep_for(std::chrono::milliseconds(pauseTime));
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    fprintf(stderr, "Time: %lf. MPI rank: %d. Process ID: %d. Thread index: %d. pauseTime = %d ms \n", MPI_Wtime(), rank, getpid(), tnum, pauseTime);    
}

extern "C"
void testThreads(int rank){    
    std::vector<std::thread> threads;
	for(int i = 0; i < std::thread::hardware_concurrency()-1; i++) {
		std::thread thr(thread_proc, i, rank);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}    
}