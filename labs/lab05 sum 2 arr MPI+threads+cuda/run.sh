git pull

cat description.md
echo ''
echo '------- Compiling main.c into main.o: mpicxx -O3 -c main.cpp ------'
mpicxx -O3 -std=c++11 -c main.cpp
mpicxx -O3 -std=c++11 -c utils.cpp
mpicxx -O3 -std=c++11 -c cpuThreads.cpp
echo '------- Compiling gpu.cu into gpu.o ------'
nvcc -arch sm_70 --ptxas-options=-v  -std=c++11 -c gpu.cu
echo '-------- mpicxx -o myapp4 main.o utils.o cpuThreads.o gpu.o-----'
mpicxx -L/usr/local/cuda/lib64 -lcudart -lm -o myapp4 main.o utils.o cpuThreads.o gpu.o
echo '-------------'

#echo '------- Starting myapp4 in 1 node with 1 cpu per node with maxtime by 2 minutes ------'
echo '------- mpirun -np 1 -ppn 1 -maxtime 2 ./myapp4 --------'
mpirun -np 1 -ppn 1 -maxtime 2 ./myapp4

#echo '------- mpirun -np 2 -ppn 1 -maxtime 2 ./myapp4 --------'
#mpirun -np 2 -ppn 1 -maxtime 2 ./myapp4

echo '------- Task list: mps ---------'
mps

echo '------- Task list: pult t ---------'
pult t

echo '------- Status of executed task: mqtest myapp4.1 --------'
mqtest myapp4.1
#mqtest myapp4.2
#mqtest myapp4.3
echo '------- End of run.sh ------'