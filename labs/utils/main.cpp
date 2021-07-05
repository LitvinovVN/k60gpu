#include <iostream> 
#include <thread>
#include <vector>

#include "utils.h"

using namespace std;

void thread_proc(int tnum) {
    fprintf(stderr, "Thread %d started... \n", tnum );

    auto start = std::chrono::system_clock::now();
    int pauseTime = tnum;
    std::this_thread::sleep_for(std::chrono::seconds(pauseTime));
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    fprintf(stderr, "Process ID: %d. Thread index: %d. pauseTime = %d \n", getpid(), tnum, pauseTime);
}



int main (int argc, char* argv[])
{        
    app_init(argc, argv);     
    
    printTime("Time point 1");

    std::vector<std::thread> threads;
	for(int i = 0; i < 5+std::thread::hardware_concurrency()-1; i++) {
		std::thread thr(thread_proc, i);
		threads.emplace_back(std::move(thr));
	}
	
	for(auto& thr : threads) {
		thr.join();
	}

    printTime("Time point 2");
    
    return 0;
}