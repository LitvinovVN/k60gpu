#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <thread>

#include "utils.h"

void mpi_init(int argc, char* argv[], int mpi_thread_type, int &provided, int &rank, int &size){
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    double t = MPI_Wtime();
    char hostname[50];    
    gethostname(hostname, 50);
    pid_t pid = getpid();
    int nHardwareThreads = std::thread::hardware_concurrency();
    fprintf(stderr, "Time: %lf. Hostname: %s. MPI rank: %d. Process ID: %d. Hardware threads: %d \n", t, hostname, rank, pid, nHardwareThreads);

    if(rank==0) {
        printf("argc: %d\n", argc);
        printf("&argc: %p\n", (void*)&argc);
        
        for(int i=0;i<argc;i++)
        {
            printf("argv[%d]: %s\n", i, argv[i]);
        }

        printf("mpi_thread_type: %d\n", mpi_thread_type);

        printf("provided: %d\n",   provided);
        printf("&provided: %p\n", &provided);

        printf("size: %d\n",   size);
        printf("&size: %p\n", &size);

        if(provided < MPI_THREAD_FUNNELED)
        {
            printf("The threading support level is lesser than that demanded.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        else
        {
            printf("The threading support level: MPI_THREAD_FUNNELED\n");
        }
    }        

    printf("rank: %d\n",   rank);
    printf("&rank: %p\n", &rank);    
    

    printf("hardware_concurrency(): %d\n", std::thread::hardware_concurrency());
}