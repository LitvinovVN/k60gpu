#include "head.h"

int main(int argc, char* argv[]){
	
	int rank, size;
	int x = 25;
	int y = 20;
	
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);//номер текущего процесса
    MPI_Comm_size (MPI_COMM_WORLD, &size);//число процессов
	
	int res_gpu = gpu(x, y);	
	int res_cpu = cpu(x, y);
	
	std::cout<<"res_gpu = "<<res_gpu<<std::endl;
	std::cout<<"res_cpu = "<<res_cpu<<std::endl;
	
	MPI_Finalize();
	return 0;
}