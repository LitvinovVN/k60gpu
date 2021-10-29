cmake_minimum_required (VERSION 2.8)

project ("CUDA_CMAKE" LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

cuda_select_nvcc_arch_flags(ARCH_FLAGS) # optional argument for arch to add

# Добавьте источник в исполняемый файл этого проекта.
cuda_add_executable (CUDA_CMAKE "CUDA_CMAKE.cu" "CUDA_CMAKE.h")

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
