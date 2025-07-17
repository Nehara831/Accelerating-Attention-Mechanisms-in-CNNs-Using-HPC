#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include <vector>
#include <string>

// Type alias for matrix (vector of vectors)
using Matrix = std::vector<std::vector<float>>;

// CPU attention implementation
Matrix attention_cpu(const Matrix& Q, const Matrix& K, const Matrix& V);

// Matrix utilities
Matrix transpose(const Matrix& mat);
Matrix multiply(const Matrix& A, const Matrix& B);
Matrix softmax(const Matrix& mat);
Matrix scale(const Matrix& mat, float scalar);
Matrix add(const Matrix& A, const Matrix& B);

// Utility functions
void print_matrix(const Matrix& mat, const std::string& name, int max_rows = 5, int max_cols = 5);
bool validate_matrices(const Matrix& Q, const Matrix& K, const Matrix& V);

// Multi-head and masked attention (CPU versions)
Matrix multi_head_attention_cpu(const Matrix& Q, const Matrix& K, const Matrix& V, int num_heads);
Matrix masked_attention_cpu(const Matrix& Q, const Matrix& K, const Matrix& V, bool causal_mask = false);

#endif // ATTENTION_HPP