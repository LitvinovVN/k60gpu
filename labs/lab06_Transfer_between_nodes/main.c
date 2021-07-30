#include <mpi.h>
#include <iostream> 
#include <unistd.h>
#include <malloc.h>
#include <math.h>
#include <cmath>

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
    
    
    
    double* data;
    double dtime[100];
    int numElements;
    MPI_Status status;
    int tag0 = 0;
    int k;

    for(numElements = 10000; numElements <= 100000; numElements+=10000)
    {
        for(k = 0; k < 100; k++)
        {
            int dataSize = numElements * sizeof(double);
            data = (double*)malloc(dataSize);

            // Инициализируем массив на узле 0
            if (rank == 0)
            {
                for(int i = 0; i<numElements; i++)
                {
                    data[i] = i;                
                }
            }        

            MPI_Barrier(MPI_COMM_WORLD);                

            double tStart = MPI_Wtime();
            if(rank==0)
            {
                MPI_Send(data, numElements, MPI_DOUBLE, 1, tag0, MPI_COMM_WORLD);
            }
            if(rank==1)
            {
                MPI_Recv(data, numElements, MPI_DOUBLE, 0, tag0, MPI_COMM_WORLD, &status);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double tEnd = MPI_Wtime();

            // Получение экспериментальных данных: вычисление времени работы для каждого numElements, сохранение результатов в массив dtime
            dtime[k] = tEnd - tStart;

            if(rank==0)
            {
                fprintf(stderr, "%d %d %lf\n", numElements, k, dtime[k]);
            }
             
        }

        if(rank==0)
        {
            double Sum = 0;
            // Обработка результатов эксперимента
            
            for(k = 0; k < 100; k++)
            {
                // Вычисление суммы элементов
                Sum = Sum + dtime[k];

                // Упорядочивание элементов массива по возрастанию
                    for(int j=k; j<100; j++)
                {
                    if(dtime[k]>dtime[j])
                    {
                    int temp=dtime[k];
                    dtime[k]=dtime[j];
                    dtime[j]=temp;
                    }
                }           
                                          

            }

            for(k = 0; k < 100; k++)
            {
                fprintf(stderr, "k=%d, %lf\n", k, dtime[k]);
            }
            

            double Min = dtime[99];
            double Max = dtime[0];
            double Perc95 = dtime[5];

            // Вычисление среднего значения AvgDtime
            double AvgDtime = Sum/100;

            // Вычисление дисперсии Variance
            double VarSum = 0;
            
            for(k = 0; k < 100; k++)
            {
                VarSum = VarSum + (dtime[k] - AvgDtime);
            }
            
            double Variance = VarSum / 99;

            // Вычисление среднего квадратичного отклонения StdDev
            double StdDev = sqrt(abs(Variance));

            fprintf(stderr, "numElements=%d, AvgDtime=%lf, Min=%lf, Max=%lf, Perc95=%lf, Variance=%lf, StdDev=%lf\n", numElements, AvgDtime, Min, Max, Perc95, Variance, StdDev);
                
            }
        
      
    }

        

    /*for(int i = 0; i<numElements; i++)
    {            
        fprintf(stderr, "Node: %d. data[%d] %lf. \n", rank, i, data[i]);
    }*/

    MPI_Finalize();
    return 0;
}