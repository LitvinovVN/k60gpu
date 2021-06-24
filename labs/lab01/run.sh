git pull

cat description.md

echo '------- Compiling main.c into main.o ------'
mpicxx -O3 -c main.c
echo '-------------'
mpicxx -o myapp main.o
echo '-------------'

mpirun -np 2 -ppn 1 -maxtime 60 ./myapp