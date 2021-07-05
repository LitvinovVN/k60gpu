git pull

cat description.md
echo ''
echo '------- Compiling main.c into main.o: mpicxx -O3 -c main.cpp ------'
mpicxx -O3 -std=c++11 -c main.cpp
echo '-------- mpicxx -o myapp main.o -----'
mpicxx -o myapp2 main.o
echo '-------------'

#echo '------- Starting myapp in 1 node with 4 cpu per node with maxtime by default (5 minutes) ------'
#echo '------- mpirun -np 4 ./myapp --------'
#mpirun -np 4 ./myapp

echo '------- mpirun -np 1 -ppn 1 -maxtime 2 ./myapp --------'
mpirun -np 1 -ppn 1 -maxtime 2 ./myapp2

#echo '------- mpirun -np 2 -ppn 1 -maxtime 2 ./myapp --------'
#mpirun -np 2 -ppn 1 -maxtime 2 ./myapp

echo '------- Task list: mps ---------'
mps

echo '------- Task list: pult t ---------'
pult t

echo '------- Status of executed task: mqtest myapp.1 --------'
mqtest myapp2.1
#mqtest myapp.2
#mqtest myapp.3
echo '------- End of run.sh ------'