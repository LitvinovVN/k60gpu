echo '------ Updating git-repository -------'
git pull

cat description.md
echo ''
echo '------- Compiling gpu.cu into gpu.o ------'
nvcc --compiler-options -O3 -arch sm_70 --ptxas-options=-v -c gpu.cu
echo '-------------'
echo '------- Compiling cpu.cpp into cpu.o ------'
mpicxx -O3 -c cpu.cpp
echo '-------------'
echo '------- Compiling main.cpp into main.o: mpicxx -O3 -std=c++11 -c main.cpp ------'
mpicxx -O3 -std=c++11 -c main.cpp
echo '------ Creating executable file myapp -------'
mpicxx -L/usr/local/cuda/lib64 -lcudart -lm  -o myapp3 main.o cpu.o gpu.o
echo '-------------'

echo '------- Starting myapp in 1 node with 4 cpu per node with maxtime by default (5 minutes) ------'
echo '------- mpirun -np 4 ./myapp --------'
#mpirun -np 4 ./myapp

echo '------- mpirun -np 1 -ppn 1 -maxtime 2 ./myapp --------'
mpirun -np 1 -ppn 1 -maxtime 2 ./myapp3

echo '------- mpirun -np 2 -ppn 1 -maxtime 2 ./myapp --------'
#mpirun -np 2 -ppn 1 -maxtime 2 ./myapp

echo '------- Task list: mps ---------'
mps

echo '------- Task list: pult t ---------'
pult t

echo '------- Status of executed task: mqtest myapp.1 --------'
mqtest myapp3.1
#mqtest myapp.2
#mqtest myapp.3
echo '------- End of run.sh ------'