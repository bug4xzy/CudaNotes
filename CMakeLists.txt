cmake_minimum_required(VERSION 3.10)

# 设置 CUDA 编译器和主机编译器
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/bin/nvcc")
endif()

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")
endif()


project(CudaNote CUDA CXX)

find_package(GTest REQUIRED)


# CUDA设置
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)  # 根据你的GPU架构调整

# 设置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# 添加子目录
add_subdirectory(note0_vectoradd)
add_subdirectory(note2_reduction)
enable_testing()
