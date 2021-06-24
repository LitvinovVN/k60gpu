git pull

cat description.md
echo ''
echo '------- Compiling main.c into main.o: mpicxx -O3 -c main.c ------'
mpicxx -O3 -c main.c
echo '-------- mpicxx -o myapp main.o -----'
mpicxx -o myapp main.o
echo '-------------'

echo '------- Starting myapp in 1 node with 4 cpu per node with maxtime 60 minutes ------'
echo '------- mpirun -np 4 -maxtime 60 ./myapp --------'
mpirun -np 4 ./myapp

echo '------- mpirun -np 1 -ppn 1 -maxtime 60 ./myapp --------'
mpirun -np 1 -ppn 1 -maxtime 2 ./myapp

echo '------- mpirun -np 2 -ppn 1 -maxtime 60 ./myapp --------'
mpirun -np 2 -ppn 1 -maxtime 2 ./myapp

echo '------- Task list: mps ---------'
mps

echo '------- Task list: pult t ---------'
pult t

echo '------- Status of executed task: mqtest myapp.1 --------'
mqtest myapp.1
mqtest myapp.2
mqtest myapp.3
echo '------- End of run.sh ------'