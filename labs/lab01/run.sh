git pull

cat description.md

echo '------- Compiling main.c into main.o: mpicxx -O3 -c main.c ------'
mpicxx -O3 -c main.c
echo '-------- mpicxx -o myapp main.o -----'
mpicxx -o myapp main.o
echo '-------------'

echo '------- Starting myapp in 1 node with 1 cpu per node with maxtime 60 minutes ------'
echo 'mpirun -np 1 -ppn 1 -maxtime 60 ./myapp'
mpirun -np 1 -ppn 1 -maxtime 60 ./myapp

echo '------- Status of executed task: mqtest myapp.1 --------'
mqtest myapp.1
echo '------- End of run.sh ------'