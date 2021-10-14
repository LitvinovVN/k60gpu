project(test_cmake)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

add_executable(test_cmake main.cpp "tests/testFunctions.cpp")
