#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <thread>
#include <vector>
#include <chrono>

#include "utils.h"

using namespace std;

struct Stopwatch {
    Stopwatch(/*std::chrono::nanoseconds& result*/)
     : /*_result{result},*/ _start{ std::chrono::high_resolution_clock::now() }
    {
        //_result = result;
        //start = std::chrono::high_resolution_clock::now();
    }

    ~Stopwatch() {
         _result = std::chrono::high_resolution_clock::now() - _start;
    }

private:
    std::chrono::nanoseconds _result;
    const std::chrono::time_point<std::chrono::high_resolution_clock> _start;
};


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
    std::vector<std::thread> threads;
	for(int i = 0; i < std::thread::hardware_concurrency()-1; i++) {
		std::thread thr(thread_proc, i, rank);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}
}

int main (int argc, char* argv[])
{    
    int rank, size, provided;    
    mpi_init(argc, argv, MPI_THREAD_FUNNELED, provided, rank, size);     
    
    testThreads(rank);


    /*const size_t n = 1'000'000;
    std::chrono::nanoseconds elapsed;
    {
        Stopwatch stopwatch{elapsed};
        volatile double result{1.23e45};
        for(double i=1; i<n; i++)
        {
            result /= i;
        }
    }
    auto time_per_division = elapsed.count() / double{n};
    printf("Took %g ns per division.\n", time_per_division);*/
    
    MPI_Finalize();
    return 0;
}