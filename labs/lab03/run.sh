git pull

cat description.md
echo ''
echo '------- Compiling main.c into main.o: mpicxx -O3 -c main.cpp ------'
mpicxx -O3 -std=c++11 -c main.cpp
mpicxx -O3 -std=c++11 -c utils.cpp
mpicxx -O3 -std=c++11 -c cpuThreads.cpp
echo '------- Compiling gpu.cu into gpu.o ------'
nvcc -arch sm_70 --ptxas-options=-v -c gpu.cu
echo '-------- mpicxx -o myapp3 main.o utils.o cpuThreads.o-----'
mpicxx -L/usr/local/cuda/lib64 -lcudart -lm -o myapp3 main.o utils.o cpuThreads.o gpu.o
echo '-------------'

#echo '------- Starting myapp3 in 1 node with 1 cpu per node with maxtime by 2 minutes ------'
echo '------- mpirun -np 1 -ppn 1 -maxtime 2 ./myapp3 --------'
mpirun -np 1 -ppn 1 -maxtime 2 ./myapp3

#echo '------- mpirun -np 2 -ppn 1 -maxtime 2 ./myapp --------'
#mpirun -np 2 -ppn 1 -maxtime 2 ./myapp

echo '------- Task list: mps ---------'
mps

echo '------- Task list: pult t ---------'
pult t

echo '------- Status of executed task: mqtest myapp3.1 --------'
mqtest myapp3.1
#mqtest myapp3.2
#mqtest myapp3.3
echo '------- End of run.sh ------'