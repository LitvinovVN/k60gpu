mkdir _build
cd _build
cmake ..  -DCMAKE_BUILD_TYPE=Release
cmake --build .

cd Debug
CUDA_CMAKE.exe