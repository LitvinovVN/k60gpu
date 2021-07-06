#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <thread>
#include <vector>
#include <chrono> // chrono::system_clock
#include <ctime>   // localtime
#include <iomanip> // put_time

#include "utils.h"

using namespace std;

void printTime(std::string timePointDescr){
    auto now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);  

    std::cout << std::put_time(std::localtime(&tt), "%X") << ": " << timePointDescr << std::endl;
    std::cerr << "errors";
}

void thread_proc(int tnum, int rank) {
    fprintf(stderr, "Thread %d started at node %d... \n", tnum, rank);

    auto start = std::chrono::system_clock::now();
    int pauseTime = tnum;
    std::this_thread::sleep_for(std::chrono::seconds(pauseTime));
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    fprintf(stderr, "Time: %lf. MPI rank: %d. Process ID: %d. Thread index: %d. pauseTime = %d \n", MPI_Wtime(), rank, getpid(), tnum, pauseTime);    
}

void testThreads(int rank){
    printTime("testThreads started.\n");
    std::vector<std::thread> threads;
	for(int i = 0; i < std::thread::hardware_concurrency()-1; i++) {
		std::thread thr(thread_proc, i, rank);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}
    printTime("testThreads ended.\n");
}

int main (int argc, char* argv[])
{    
    int rank, size, provided;    
    mpi_init(argc, argv, MPI_THREAD_FUNNELED, provided, rank, size);     
    
    testThreads(rank);   
    
    MPI_Finalize();
    return 0;
}