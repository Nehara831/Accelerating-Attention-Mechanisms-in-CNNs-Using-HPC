//attention_cpu.cpp
#include "attention.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <ctime>

// Matrix transpose
Matrix transpose(const Matrix& mat) {
    if (mat.empty() || mat[0].empty()) {
        throw std::invalid_argument("Cannot transpose empty matrix");
    }
    
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(cols, std::vector<float>(rows));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = mat[i][j];
        }
    }
    
    return result;
}

// Matrix multiplication
Matrix multiply(const Matrix& A, const Matrix& B) {
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
    
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            for (int k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}

// Softmax function applied row-wise
Matrix softmax(const Matrix& mat) {
    if (mat.empty() || mat[0].empty()) {
        throw std::invalid_argument("Cannot apply softmax to empty matrix");
    }
    
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(rows, std::vector<float>(cols));
    
    for (int i = 0; i < rows; ++i) {
        float max_val = *std::max_element(mat[i].begin(), mat[i].end());
        
        //  exp(x - max) and sum
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

// Scale matrix by a scalar
Matrix scale(const Matrix& mat, float scalar) {
    if (mat.empty() || mat[0].empty()) {
        return mat;
    }
    
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(rows, std::vector<float>(cols));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = mat[i][j] * scalar;
        }
    }
    
    return result;
}

// Add two matrices element-wise
Matrix add(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
        throw std::invalid_argument("Cannot add empty matrices");
    }
    
    int A_rows = A.size();
    int A_cols = A[0].size();
    int B_rows = B.size();
    int B_cols = B[0].size();
    
    if (A_rows != B_rows || A_cols != B_cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(A_rows, std::vector<float>(A_cols));
    
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < A_cols; ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    
    return result;
}

void print_matrix(const Matrix& mat, const std::string& name, int max_rows, int max_cols) {
    if (mat.empty()) {
        std::cout << name << ": [empty matrix]" << std::endl;
        return;
    }
    
    int rows = std::min(static_cast<int>(mat.size()), max_rows);
    int cols = std::min(static_cast<int>(mat[0].size()), max_cols);
    
    std::cout << name << " (" << mat.size() << "x" << mat[0].size() << "):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << mat[i][j] << " ";
        }
        if (mat[0].size() > max_cols) {
            std::cout << "...";
        }
        std::cout << std::endl;
    }
    if (mat.size() > max_rows) {
        std::cout << "  ..." << std::endl;
    }
    std::cout << std::endl;
}

// Validate matrix dimensions
bool validate_matrices(const Matrix& Q, const Matrix& K, const Matrix& V) {
    if (Q.empty() || K.empty() || V.empty()) {
        std::cerr << "Error: One or more input matrices are empty" << std::endl;
        return false;
    }
    
    if (Q[0].empty() || K[0].empty() || V[0].empty()) {
        std::cerr << "Error: One or more input matrices have empty rows" << std::endl;
        return false;
    }
    
    int Q_rows = Q.size(), Q_cols = Q[0].size();
    int K_rows = K.size(), K_cols = K[0].size();
    int V_rows = V.size(), V_cols = V[0].size();
    
    if (Q_cols != K_cols) {
        std::cerr << "Error: Q and K must have same number of columns (embed_dim)" << std::endl;
        std::cerr << "Q: " << Q_rows << "x" << Q_cols << ", K: " << K_rows << "x" << K_cols << std::endl;
        return false;
    }
    
    if (K_rows != V_rows) {
        std::cerr << "Error: K and V must have same number of rows (seq_len)" << std::endl;
        std::cerr << "K: " << K_rows << "x" << K_cols << ", V: " << V_rows << "x" << V_cols << std::endl;
        return false;
    }
    
    for (const auto& row : Q) {
        if (row.size() != Q_cols) {
            std::cerr << "Error: Inconsistent row sizes in Q matrix" << std::endl;
            return false;
        }
    }
    
    for (const auto& row : K) {
        if (row.size() != K_cols) {
            std::cerr << "Error: Inconsistent row sizes in K matrix" << std::endl;
            return false;
        }
    }
    
    for (const auto& row : V) {
        if (row.size() != V_cols) {
            std::cerr << "Error: Inconsistent row sizes in V matrix" << std::endl;
            return false;
        }
    }
    
    return true;
}

// CPU attention implementation
Matrix attention_cpu(const Matrix& Q, const Matrix& K, const Matrix& V) {
    if (!validate_matrices(Q, K, V)) {
        throw std::invalid_argument("Invalid matrix dimensions for attention computation");
    }
    
    int seq_len_q = Q.size();
    int embed_dim = Q[0].size();
    int seq_len_k = K.size();
    int embed_dim_v = V[0].size();
    
    try {
        // Q * K^T
        Matrix K_T = transpose(K);
        Matrix scores = multiply(Q, K_T);
        
        float scale_factor = 1.0f / std::sqrt(static_cast<float>(embed_dim));
        scores = scale(scores, scale_factor);
        
        Matrix attention_weights = softmax(scores);
        
        Matrix output = multiply(attention_weights, V);
        
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in CPU attention computation: " << e.what() << std::endl;
        throw;
    }
}
