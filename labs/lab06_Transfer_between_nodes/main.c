#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <malloc.h>

using namespace std;

int main (int argc, char* argv[])
{
    int rank, size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    if(rank==0)
        cout << "Node-Node data transfer test." << " Size is " << size << endl;

    double t = MPI_Wtime();

    char hostname[50];    
    gethostname(hostname, 50);

    pid_t pid = getpid();
        
    fprintf(stderr, "Time: %lf. Hostname: %s. MPI rank: %d. Process ID: %d. \n", t, hostname, rank, pid);
    
    // 1. Создаём массив
    double* data;
    int numElements = 100;
    int dataSize = numElements * sizeof(double);
    data = (double*)malloc(dataSize);

    // Инициализируем массив на узле 0
    if (rank == 0)
    {
        for(int i = 0; i<numElements; i++)
        {
            data[i] = i;
            fprintf(stderr, "Node: %d. data[%d] %lf. \n", rank, i, data[i]);
        }
    }
    
    /*for(int i = 0; i<numElements; i++)
    {            
        fprintf(stderr, "Node: %d. data[%d] %lf. \n", rank, i, data[i]);
    }*/

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status status;
    int tag0 = 0;
    //MPI_Sendrecv(data, numElements, MPI_DOUBLE, 1, 10, data, numElements, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &status);
    
    if(rank==0)
    {
        MPI_Send(data, numElements, MPI_DOUBLE, 1, tag0, MPI_COMM_WORLD);
    }
    if(rank==1)
    {
        MPI_Recv(data, numElements, MPI_DOUBLE, 0, tag0, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = 0; i<numElements; i++)
    {            
        fprintf(stderr, "Node: %d. data[%d] %lf. \n", rank, i, data[i]);
    }

    MPI_Finalize();
    return 0;
}