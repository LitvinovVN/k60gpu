git pull

mkdir _build
cd _build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

./cpp11