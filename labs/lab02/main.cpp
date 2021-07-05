#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <thread>
#include <vector>

using namespace std;

void thread_proc(int tnum, char* hostname, int rank) {
    fprintf(stderr, "Thread %d started... \n", tnum );

    auto start = std::chrono::system_clock::now();
    int pauseTime = tnum;
    std::this_thread::sleep_for(std::chrono::seconds(pauseTime));
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    fprintf(stderr, "Time: %lf. Hostname: %s. MPI rank: %d. Process ID: %d. Thread index: %d. pauseTime = %d \n", MPI_Wtime(), hostname, rank, getpid(), tnum, pauseTime);
}


void mpi_init(int argc/*, char* argv[], int provided, int rank, int size*/){
    printf("argc: %d\n", argc);
    printf("&argc: %p\n", (void*)argc);
}


int main (int argc, char* argv[])
{    
    int rank, size, provided;
    mpi_init(argc/*, argv, MPI_THREAD_FUNNELED, &provided, &rank, &size*/);


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
            printf("The threading support level: MPI_THREAD_FUNNELED\n");
        }

        printf("hardware_concurrency(): %d\n", std::thread::hardware_concurrency());
    }    

    
    
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

    MPI_Finalize();
    return 0;
}