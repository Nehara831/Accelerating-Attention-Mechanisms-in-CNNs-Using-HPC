cmake_minimum_required(VERSION 3.10)
project(attention_cuda_py LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(attention_cuda_py
    attention_cuda.cu
    attention.cpp
    python_binding.cpp
)

# Use GPU architectures supported by CUDA 10.1
set_property(TARGET attention_cuda_py PROPERTY CUDA_ARCHITECTURES "60;70;75")
target_link_libraries(attention_cuda_py PRIVATE ${CUDA_LIBRARIES})