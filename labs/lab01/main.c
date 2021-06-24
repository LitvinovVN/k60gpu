#include <mpi.h>
#include <iostream> 
#include <unistd.h>

using namespace std;

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