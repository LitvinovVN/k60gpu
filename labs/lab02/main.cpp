#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <atomic>
#include <vector>
#include <thread>

using namespace std;



class thread_pool 
{
    std::atomic_bool done;
    threadsafe_queue<std::function<void()>> work_queue;
    std::vector<std::thread> threads;
    join_threads joiner;
    void worker_thread()
    {
        while(!done)
        {
            std::function<void()> task; 
            if(work_queue.try_pop(task))
            {
                task();
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }
public:
    thread_pool():
    done(false), joiner(threads)
    {
        unsigned const thread_count = std::thread::hardware_concurrency();
        try 
        {
            for(unsigned i=0; i<thread_count; ++i)
            {
                threads.push_back(std::thread(&thread_pool::worker_thread, this));
            }
        }
        catch (...)
        {
            done = true;
            throw;
        }
    }

    ~thread_pool()
    {
        done = true;
    }

    template<typename FunctionType> 
    void submit(FunctionType f)
    {
        work_queue.push(std::function< void()> (f));
    }
};


int main (int argc, char* argv[])
{
    int rank, size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    if(rank==0)
        cout << "First MPI program." << " Size is " << size << endl;

    double t = MPI_Wtime();

    char hostname[50];    
    gethostname(hostname, 50);

    pid_t pid = getpid();

        
    fprintf(stderr, "Time: %lf. Hostname: %s. MPI rank: %d. Process ID: %d. \n", t, hostname, rank, pid);
    
    MPI_Finalize();
    return 0;
}