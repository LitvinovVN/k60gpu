clear

cat description.md
echo '-------------'

echo '------ Updating git-repository -------'
git pull

echo '------ Removing temporary files -------'
rm -r myapp.1
rm *.o myapp
echo '-------------'

echo '------- Compiling gpu.cu into gpu.o ------'
nvcc --compiler-options -O3 -arch sm_70 --ptxas-options=-v -c gpu.cu
echo '-------------'

echo '------ mpicxx -show -------'
mpicxx -show
echo '-------------'
echo '------- Compiling cpu.cpp into cpu.o ------'
mpicxx -O3 -c cpu.cpp
echo '-------------'
echo '------- Compiling main.cpp into main.o ------'
mpicxx -O3 -std=c++11 -c main.cpp
echo '-------------'
echo '------ Creating executable file myapp -------'
mpicxx -L/usr/local/cuda/lib64 -lcudart -lm  -o myapp main.o cpu.o gpu.o
echo '-------------'

echo '------ Starting myapp -------'
mpirun -np 8 ./myapp

echo '----- Sleeping 5 seconds -------'
sleep 5

echo '-------------------'
echo '-----RESULTS-------'
echo '-------------------'
cat ./myapp.1/output