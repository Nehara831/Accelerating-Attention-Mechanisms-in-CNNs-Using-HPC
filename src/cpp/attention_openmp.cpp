#include "attention_openmp.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>

// OpenMP matrix transpose
Matrix transpose_openmp(const Matrix& mat) {
    if (mat.empty() || mat[0].empty()) {
        throw std::invalid_argument("Cannot transpose empty matrix");
    }
    
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(cols, std::vector<float>(rows));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = mat[i][j];
        }
    }
    
    return result;
}

// OpenMP matrix multiplication
Matrix multiply_openmp(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
        throw std::invalid_argument("Cannot multiply empty matrices");
    }
    
    int A_rows = A.size();
    int A_cols = A[0].size();
    int B_rows = B.size();
    int B_cols = B[0].size();
    
    if (A_cols != B_rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(A_rows, std::vector<float>(B_cols, 0.0f));
    
    #pragma omp parallel for
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            for (int k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}

// OpenMP softmax
Matrix softmax_openmp(const Matrix& mat) {
    if (mat.empty() || mat[0].empty()) {
        throw std::invalid_argument("Cannot apply softmax to empty matrix");
    }
    
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(rows, std::vector<float>(cols));
    
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        // Find max value in the row for numerical stability
        float max_val = mat[i][0];
        for (int j = 1; j < cols; ++j) {
            max_val = std::max(max_val, mat[i][j]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            result[i][j] = std::exp(mat[i][j] - max_val);
            sum += result[i][j];
        }
        
        // Normalize
        for (int j = 0; j < cols; ++j) {
            result[i][j] /= sum;
        }
    }
    
    return result;
}

// OpenMP matrix scaling
Matrix scale_openmp(const Matrix& mat, float scalar) {
    if (mat.empty() || mat[0].empty()) {
        return mat;
    }
    
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(rows, std::vector<float>(cols));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = mat[i][j] * scalar;
        }
    }
    
    return result;
}

// Main OpenMP attention implementation
Matrix attention_openmp(const Matrix& Q, const Matrix& K, const Matrix& V) {
    if (Q.empty() || K.empty() || V.empty()) {
        throw std::invalid_argument("Input matrices cannot be empty");
    }
    
    if (Q[0].empty() || K[0].empty() || V[0].empty()) {
        throw std::invalid_argument("Input matrices cannot have empty rows");
    }
    
    int Q_rows = Q.size(), Q_cols = Q[0].size();
    int K_rows = K.size(), K_cols = K[0].size();
    int V_rows = V.size(), V_cols = V[0].size();
    
    if (Q_cols != K_cols) {
        throw std::invalid_argument("Q and K must have same number of columns (embed_dim)");
    }
    
    if (K_rows != V_rows) {
        throw std::invalid_argument("K and V must have same number of rows (seq_len)");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Step 1: Compute Q * K^T using OpenMP
    Matrix K_T = transpose_openmp(K);
    Matrix scores = multiply_openmp(Q, K_T);
    
    // Step 2: Scale by 1/sqrt(d_k)
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(Q_cols));
    scores = scale_openmp(scores, scale_factor);
    
    // Step 3: Apply softmax
    Matrix attention_weights = softmax_openmp(scores);
    
    // Step 4: Multiply by V
    Matrix output = multiply_openmp(attention_weights, V);
    
    return output;
}


// OpenMP utility functions
bool openmp_available() {
    return true;
}

int get_openmp_max_threads() {
    return omp_get_max_threads();
}

void print_openmp_info() {
    std::cout << "OpenMP Information:" << std::endl;
    std::cout << "  Version: " << _OPENMP << std::endl;
    std::cout << "  Max threads: " << omp_get_max_threads() << std::endl;
    std::cout << "  Number of processors: " << omp_get_num_procs() << std::endl;
    std::cout << "  Dynamic adjustment: " << (omp_get_dynamic() ? "enabled" : "disabled") << std::endl;
    std::cout << "  Nested parallelism: " << (omp_get_nested() ? "enabled" : "disabled") << std::endl;
}

void set_openmp_threads(int num_threads) {
    if (num_threads > 0 && num_threads <= omp_get_num_procs()) {
        omp_set_num_threads(num_threads);
        std::cout << "Set OpenMP threads to: " << num_threads << std::endl;
    } else {
        std::cerr << "Warning: Invalid thread count " << num_threads 
                  << ". Using default (" << omp_get_max_threads() << ")" << std::endl;
    }
}