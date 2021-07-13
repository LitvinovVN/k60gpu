git pull

cat description.md
echo ''
echo '------- Compiling main.c into main.o: mpicxx -O3 -c main.cpp ------'
mpicxx -O3 -std=c++11 -c main.cpp
mpicxx -O3 -std=c++11 -c utils.cpp
echo '------- Compiling sum.cu into sum.o ------'
nvcc -arch sm_70 --ptxas-options=-v  -std=c++11 -c sum.cu
echo '-------- mpicxx -L/usr/local/cuda/lib64 -lcudart -lm -o myapp5 main.o utils.o sum.o -----'
mpicxx -L/usr/local/cuda/lib64 -lcudart -lm -o myapp5 main.o utils.o sum.o
echo '-------------'

echo '------- Starting myapp5 in 1 node with 1 cpu per node with maxtime by 2 minutes ------'
echo '------- mpirun -np 1 -ppn 1 -maxtime 2 ./myapp5 --------'
mpirun -np 1 -ppn 1 -maxtime 2 ./myapp5

echo '------- Task list: mps ---------'
mps

echo '------- Task list: pult t ---------'
pult t

echo '------- Status of executed task: mqtest myapp5.1 --------'
mqtest myapp5.1
echo '------- End of run.sh ------'