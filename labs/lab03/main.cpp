#include "head.h"


std::mutex mtx;
static int counter = 0;
static const int MAX_COUNTER_VAL = 100;

void thread_proc(int tnum) {
    for(;;) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if(counter == MAX_COUNTER_VAL)
                break;
            int ctr_val = ++counter;
            std::cout << "Thread " << tnum << ": counter = " <<
                         ctr_val << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main(int argc, char* argv[]){
	
	int rank, size;
	int x = 25;
	int y = 20;
	
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);//номер текущего процесса
    MPI_Comm_size (MPI_COMM_WORLD, &size);//число процессов
	
	int res_gpu = gpu(x, y);	
	int res_cpu = cpu(x, y);
	
	std::cout<<"res_gpu = "<<res_gpu<<" (rank = "<<rank<<")"<<std::endl;
	std::cout<<"res_cpu = "<<res_cpu<<" (rank = "<<rank<<")"<<std::endl;

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0)
	{
		std::vector<std::thread> threads;
		for(int i = 0; i < 10; i++) {
			std::thread thr(thread_proc, i);
			threads.emplace_back(std::move(thr));
		}

		// can't use const auto& here since .join() is not marked const
		for(auto& thr : threads) {
			thr.join();
		}

		std::cout << "Done!" << std::endl;
	}

	
	MPI_Finalize();
	return 0;
}