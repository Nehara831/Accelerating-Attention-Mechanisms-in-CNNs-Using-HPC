#include "attention_openmp.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// OpenMP matrix transpose with parallel processing
Matrix transpose_openmp(const Matrix& mat) {
    if (mat.empty() || mat[0].empty()) {
        throw std::invalid_argument("Cannot transpose empty matrix");
    }
    
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(cols, std::vector<float>(rows));
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = mat[i][j];
        }
    }
#else
    // Fallback to serial implementation
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = mat[i][j];
        }
    }
#endif
    
    return result;
}

// OpenMP matrix multiplication with optimized parallel processing
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
    
#ifdef _OPENMP
    // Use collapse(2) for better load balancing on larger matrices
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            float sum = 0.0f;
            // Vectorize inner loop if possible
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < A_cols; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
#else
    // Fallback to serial implementation
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            for (int k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
#endif
    
    return result;
}

// OpenMP softmax with parallel row processing
Matrix softmax_openmp(const Matrix& mat) {
    if (mat.empty() || mat[0].empty()) {
        throw std::invalid_argument("Cannot apply softmax to empty matrix");
    }
    
    int rows = mat.size();
    int cols = mat[0].size();
    Matrix result(rows, std::vector<float>(cols));
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        // Find max value in the row for numerical stability
        float max_val = mat[i][0];
        #pragma omp simd reduction(max:max_val)
        for (int j = 1; j < cols; ++j) {
            max_val = std::max(max_val, mat[i][j]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        #pragma omp simd
        for (int j = 0; j < cols; ++j) {
            result[i][j] = std::exp(mat[i][j] - max_val);
        }
        
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < cols; ++j) {
            sum += result[i][j];
        }
        
        // Normalize
        #pragma omp simd
        for (int j = 0; j < cols; ++j) {
            result[i][j] /= sum;
        }
    }
#else
    // Fallback to serial implementation
    for (int i = 0; i < rows; ++i) {
        float max_val = *std::max_element(mat[i].begin(), mat[i].end());
        
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            result[i][j] = std::exp(mat[i][j] - max_val);
            sum += result[i][j];
        }
        
        for (int j = 0; j < cols; ++j) {
            result[i][j] /= sum;
        }
    }
#endif
    
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
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = mat[i][j] * scalar;
        }
    }
#else
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = mat[i][j] * scalar;
        }
    }
#endif
    
    return result;
}

// Main OpenMP attention implementation
Matrix attention_openmp(const Matrix& Q, const Matrix& K, const Matrix& V) {
    // Validate inputs (reuse from attention.cpp)
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
    
    try {
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
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
#ifdef _OPENMP
        std::cout << "OpenMP attention computation completed in " << duration.count() << " μs" 
                  << " using " << omp_get_max_threads() << " threads" << std::endl;
#endif
        
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in OpenMP attention computation: " << e.what() << std::endl;
        throw;
    }
}

// Multi-head attention with OpenMP
Matrix multi_head_attention_openmp(const Matrix& Q, const Matrix& K, const Matrix& V, int num_heads) {
    if (Q.empty() || K.empty() || V.empty()) {
        throw std::invalid_argument("Input matrices cannot be empty");
    }
    
    int seq_len = Q.size();
    int embed_dim = Q[0].size();
    
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("Embedding dimension must be divisible by number of heads");
    }
    
    int head_dim = embed_dim / num_heads;
    Matrix output(seq_len, std::vector<float>(embed_dim, 0.0f));
    
#ifdef _OPENMP
    // Parallelize across heads
    #pragma omp parallel for schedule(static)
    for (int head = 0; head < num_heads; ++head) {
        int start_col = head * head_dim;
        
        // Extract head-specific Q, K, V
        Matrix Q_head(seq_len, std::vector<float>(head_dim));
        Matrix K_head(K.size(), std::vector<float>(head_dim));
        Matrix V_head(V.size(), std::vector<float>(head_dim));
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                Q_head[i][j] = Q[i][start_col + j];
            }
        }
        
        for (int i = 0; i < K.size(); ++i) {
            for (int j = 0; j < head_dim; ++j) {
                K_head[i][j] = K[i][start_col + j];
                V_head[i][j] = V[i][start_col + j];
            }
        }
        
        // Compute attention for this head (using serial version to avoid nested parallelism)
        Matrix head_output = attention_cpu(Q_head, K_head, V_head);
        
        // Copy head output to final output
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                output[i][start_col + j] = head_output[i][j];
            }
        }
    }
#else
    // Fallback to serial multi-head attention
    for (int head = 0; head < num_heads; ++head) {
        int start_col = head * head_dim;
        
        Matrix Q_head(seq_len, std::vector<float>(head_dim));
        Matrix K_head(K.size(), std::vector<float>(head_dim));
        Matrix V_head(V.size(), std::vector<float>(head_dim));
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                Q_head[i][j] = Q[i][start_col + j];
            }
        }
        
        for (int i = 0; i < K.size(); ++i) {
            for (int j = 0; j < head_dim; ++j) {
                K_head[i][j] = K[i][start_col + j];
                V_head[i][j] = V[i][start_col + j];
            }
        }
        
        Matrix head_output = attention_cpu(Q_head, K_head, V_head);
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                output[i][start_col + j] = head_output[i][j];
            }
        }
    }
#endif
    
    return output;
}

// Masked attention with OpenMP
Matrix masked_attention_openmp(const Matrix& Q, const Matrix& K, const Matrix& V, bool causal_mask) {
    if (Q.empty() || K.empty() || V.empty()) {
        throw std::invalid_argument("Input matrices cannot be empty");
    }
    
    int seq_len_q = Q.size();
    int embed_dim = Q[0].size();
    int seq_len_k = K.size();
    
    // Compute scores
    Matrix K_T = transpose_openmp(K);
    Matrix scores = multiply_openmp(Q, K_T);
    
    // Scale scores
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(embed_dim));
    scores = scale_openmp(scores, scale_factor);
    
    // Apply mask if requested
    if (causal_mask) {
        const float NEG_INF = -1e9f;
        
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < seq_len_q; ++i) {
            for (int j = i + 1; j < seq_len_k; ++j) {
                scores[i][j] = NEG_INF;
            }
        }
#else
        for (int i = 0; i < seq_len_q; ++i) {
            for (int j = i + 1; j < seq_len_k; ++j) {
                scores[i][j] = NEG_INF;
            }
        }
#endif
    }
    
    // Apply softmax and compute output
    Matrix attention_weights = softmax_openmp(scores);
    Matrix output = multiply_openmp(attention_weights, V);
    
    return output;
}

// OpenMP utility functions
bool openmp_available() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

int get_openmp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void print_openmp_info() {
#ifdef _OPENMP
    std::cout << "OpenMP Information:" << std::endl;
    std::cout << "  Version: " << _OPENMP << std::endl;
    std::cout << "  Max threads: " << omp_get_max_threads() << std::endl;
    std::cout << "  Number of processors: " << omp_get_num_procs() << std::endl;
    std::cout << "  Dynamic adjustment: " << (omp_get_dynamic() ? "enabled" : "disabled") << std::endl;
    std::cout << "  Nested parallelism: " << (omp_get_nested() ? "enabled" : "disabled") << std::endl;
#else
    std::cout << "OpenMP not available in this build" << std::endl;
#endif
}

void set_openmp_threads(int num_threads) {
#ifdef _OPENMP
    if (num_threads > 0 && num_threads <= omp_get_num_procs()) {
        omp_set_num_threads(num_threads);
        std::cout << "Set OpenMP threads to: " << num_threads << std::endl;
    } else {
        std::cerr << "Warning: Invalid thread count " << num_threads 
                  << ". Using default (" << omp_get_max_threads() << ")" << std::endl;
    }
#else
    std::cout << "OpenMP not available - cannot set thread count" << std::endl;
#endif
}