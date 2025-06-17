#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "attention.hpp"
#include "attention_cuda.hpp"
#include <iostream>

namespace py = pybind11;

// Convert numpy array to Matrix
Matrix numpy_to_matrix(py::array_t<float> input) {
    py::buffer_info buf_info = input.request();
    float *ptr = static_cast<float *>(buf_info.ptr);
    
    if (buf_info.ndim != 2) {
        std::cerr << "Error: Input array must be 2-dimensional, got " << buf_info.ndim << " dimensions" << std::endl;
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    
    int rows = buf_info.shape[0];
    int cols = buf_info.shape[1];
    
    std::cerr << "Converting numpy array to matrix: " << rows << "x" << cols << std::endl;
    
    Matrix mat(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = ptr[i * cols + j];
        }
    }
    return mat;
}

// Convert Matrix to numpy array
py::array_t<float> matrix_to_numpy(const Matrix& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    
    std::cerr << "Converting matrix to numpy array: " << rows << "x" << cols << std::endl;
    
    // Create a contiguous array with the correct memory layout
    auto result = py::array_t<float>({rows, cols});
    py::buffer_info buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);
    
    // Copy data with proper memory layout
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ptr[i * cols + j] = mat[i][j];
        }
    }
    
    return result;
}

// Python wrapper functions
py::array_t<float> py_attention_cuda(py::array_t<float> Q, 
                                     py::array_t<float> K, 
                                     py::array_t<float> V) {
    std::cerr << "\n=== CUDA Attention Debug Info ===" << std::endl;
    std::cerr << "Input Q shape: " << Q.shape(0) << "x" << Q.shape(1) << std::endl;
    std::cerr << "Input K shape: " << K.shape(0) << "x" << K.shape(1) << std::endl;
    std::cerr << "Input V shape: " << V.shape(0) << "x" << V.shape(1) << std::endl;
    
    try {
        Matrix Q_mat = numpy_to_matrix(Q);
        Matrix K_mat = numpy_to_matrix(K);
        Matrix V_mat = numpy_to_matrix(V);
        
        printf( "Calling CUDA attention implementation...");
        Matrix result = attention_cuda(Q_mat, K_mat, V_mat);
        std::cerr << "CUDA attention completed successfully" << std::endl;
        
        return matrix_to_numpy(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA attention: " << e.what() << std::endl;
        throw;
    }
}

py::array_t<float> py_attention_cpu(py::array_t<float> Q, 
                                    py::array_t<float> K, 
                                    py::array_t<float> V) {
    std::cerr << "\n=== CPU Attention Debug Info ===" << std::endl;
    std::cerr << "Input Q shape: " << Q.shape(0) << "x" << Q.shape(1) << std::endl;
    std::cerr << "Input K shape: " << K.shape(0) << "x" << K.shape(1) << std::endl;
    std::cerr << "Input V shape: " << V.shape(0) << "x" << V.shape(1) << std::endl;
    
    try {
        Matrix Q_mat = numpy_to_matrix(Q);
        Matrix K_mat = numpy_to_matrix(K);
        Matrix V_mat = numpy_to_matrix(V);
        
        std::cerr << "Calling CPU attention implementation..." << std::endl;
        Matrix result = attention_cpu(Q_mat, K_mat, V_mat);
        std::cerr << "CPU attention completed successfully" << std::endl;
        
        return matrix_to_numpy(result);
    } catch (const std::exception& e) {
        std::cerr << "Error in CPU attention: " << e.what() << std::endl;
        throw;
    }
}

PYBIND11_MODULE(attention_cuda_py, m) {
    m.doc() = "CUDA Attention Implementation";
    
    m.def("attention_cuda", &py_attention_cuda, "CUDA attention computation");
    m.def("attention_cpu", &py_attention_cpu, "CPU attention computation");
    m.def("cuda_available", &cuda_available, "Check if CUDA is available");
    m.def("get_cuda_device_count", &get_cuda_device_count, "Get CUDA device count");
    m.def("print_cuda_info", &print_cuda_info, "Print CUDA device information");
}