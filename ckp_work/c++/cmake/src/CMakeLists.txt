project(test_cmake)

set(CMAKE_CXX_FLAGS "-std=c++11")

add_executable(test_cmake main.cpp "testFunctions.cpp")
