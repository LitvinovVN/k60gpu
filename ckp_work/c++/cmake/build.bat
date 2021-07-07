mkdir _build
cd _build
cmake ../src
cmake --build .

cd Debug
test_cmake.exe