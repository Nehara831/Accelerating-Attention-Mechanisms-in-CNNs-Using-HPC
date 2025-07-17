#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include <vector>
#include <string>

using Matrix = std::vector<std::vector<float>>;

// CPU attention implementation
Matrix attention_cpu(const Matrix& Q, const Matrix& K, const Matrix& V);

Matrix transpose(const Matrix& mat);
Matrix multiply(const Matrix& A, const Matrix& B);
Matrix softmax(const Matrix& mat);
Matrix scale(const Matrix& mat, float scalar);
Matrix add(const Matrix& A, const Matrix& B);

void print_matrix(const Matrix& mat, const std::string& name, int max_rows = 5, int max_cols = 5);
bool validate_matrices(const Matrix& Q, const Matrix& K, const Matrix& V);



#endif // ATTENTION_HPP