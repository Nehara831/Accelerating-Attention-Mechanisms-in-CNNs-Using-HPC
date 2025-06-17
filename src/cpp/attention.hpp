#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include <vector>

using Matrix = std::vector<std::vector<float>>;

// CPU attention implementation
Matrix attention_cpu(const Matrix& Q, const Matrix& K, const Matrix& V);

// Matrix utilities
Matrix transpose(const Matrix& mat);
Matrix multiply(const Matrix& A, const Matrix& B);
Matrix softmax(const Matrix& mat);

#endif