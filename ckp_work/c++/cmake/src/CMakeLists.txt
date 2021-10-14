project(test_cmake)

set(CMAKE_CXX_FLAGS "-std=c++11")

add_executable(test_cmake main.cpp "tests/testFunctions.cpp" "Grid3DSrc/ClassA.cpp" "Grid3DSrc/ClassB.cpp" "Grid3DSrc/Classes.cpp" "tests/testFunctions.h")
