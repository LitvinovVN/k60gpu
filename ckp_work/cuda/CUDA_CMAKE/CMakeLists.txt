cmake_minimum_required (VERSION 2.8)

project (CUDA_CMAKE)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(CUDA QUIET REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
# Pass options to NVCC
    set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_35,code=sm_35
    )


# Добавьте источник в исполняемый файл этого проекта.
cuda_add_executable (CUDA_CMAKE "CUDA_CMAKE.cu" "CUDA_CMAKE.h")

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
