#include <mpi.h>
#include <iostream> 
#include <unistd.h>

int main (int argc, char* argv[])
{
    int rank, size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    if(rank==0)
        std::cout << "Hello MPI!!!" << " Size is " << size << std::endl;

    char hostname[50];    
    gethostname(hostname, 50);    
    double t = MPI_Wtime();

	std::cout << " My rank is " << rank << "; hostname: " << hostname << "; Time: " << t << std::endl;
    MPI_Finalize();
    return 0;
}