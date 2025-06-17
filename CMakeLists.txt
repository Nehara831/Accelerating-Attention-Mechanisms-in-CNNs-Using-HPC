cmake_minimum_required(VERSION 3.18)
project(attention_cnn LANGUAGES CXX CUDA)

# Enable modern CMake policies
cmake_policy(SET CMP0146 NEW)  # Use modern CUDA approach
cmake_policy(SET CMP0148 NEW)  # Use modern Python approach

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add the CUDA attention module
pybind11_add_module(attention_cuda_py src/cpp/attention_cuda.cu)

# Set the language for the target
set_target_properties(attention_cuda_py PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75"
)

# Link against CUDA libraries
target_link_libraries(attention_cuda_py PRIVATE
    CUDA::cudart
    CUDA::cublas
    ${CMAKE_THREAD_LIBS_INIT}
)

# Set include directories
target_include_directories(attention_cuda_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)

# Set compile definitions
target_compile_definitions(attention_cuda_py PRIVATE
    CUDA_AVAILABLE
)

# Set output directory
set_target_properties(attention_cuda_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    PREFIX ""
)