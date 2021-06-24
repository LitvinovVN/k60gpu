cat description.md

git pull
bash clear.sh

echo '------ mpicxx -show -------'
mpicxx -show
echo '------- Compiling main.c into main.o ------'
mpicxx -O3 -c main.c
echo '-------------'
mpicxx -o myapp main.o
echo '-------------'

mpirun -np 8 ./myapp

echo '----- sleep 5 seconds -------'
sleep 5

echo '-------------------'
echo '-----RESULTS-------'
echo '-------------------'
cat ./myapp.1/output