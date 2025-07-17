cmake_minimum_required(VERSION 3.10)
project(attention_cuda_py LANGUAGES CXX CUDA)

# Find required packages
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP)

# Set standards
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    attention_cuda.cu
    attention.cpp
    attention_openmp.cpp
    python_binding.cpp
)

# Create the pybind11 module
pybind11_add_module(attention_cuda_py ${SOURCES})

# GPU architectures for CUDA
set_property(TARGET attention_cuda_py PROPERTY CUDA_ARCHITECTURES "60;70;75;80;86")

# Link libraries
target_link_libraries(attention_cuda_py PRIVATE ${CUDA_LIBRARIES})

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(attention_cuda_py PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(attention_cuda_py PRIVATE OPENMP_AVAILABLE)
    message(STATUS "OpenMP found and enabled")
else()
    message(WARNING "OpenMP not found - building without OpenMP support")
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(attention_cuda_py PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-march=native -mtune=native -ffast-math>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(attention_cuda_py PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-march=native -mtune=native -ffast-math>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(attention_cuda_py PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2 /fp:fast>)
endif()

# Print configuration info
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA found: ${CUDA_FOUND}")
message(STATUS "  OpenMP found: ${OpenMP_CXX_FOUND}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP version: ${OpenMP_CXX_VERSION}")
endif()