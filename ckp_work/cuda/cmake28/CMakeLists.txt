cmake_minimum_required(VERSION 2.8)

project(cpp11)

find_package(CUDA)

#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-DVERBOSE")

#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

CUDA_ADD_EXECUTABLE(cpp11 main.cpp test.h test.cu)