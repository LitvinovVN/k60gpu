git pull

cat description.md
echo ''
echo '------- Compiling main.c into main.o: mpicxx -O3 -c main.c ------'
mpicxx -O3 -c main.c
echo '-------- mpicxx -o myapp main.o -----'
mpicxx -o myapp6 main.o
echo '-------------'

echo '------- Starting myapp6 in 2 nodes with 1 mpu-thread per node with maxtime by 2 minutes ------'
echo '------- mpirun -np 2 -ppn 1 -maxtime 2 ./myapp6 --------'
mpirun -np 2 -ppn 1 -maxtime 2 ./myapp6

echo '------- Task list: mps ---------'
mps

echo '------- Task list: pult t ---------'
pult t

echo '------- Status of executed task: mqtest myapp6.1 --------'
mqtest myapp6.1
echo '------- End of run.sh ------'