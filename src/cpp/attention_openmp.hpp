#ifndef ATTENTION_OPENMP_HPP
#define ATTENTION_OPENMP_HPP

#include "attention.hpp"

// OpenMP attention implementation
Matrix attention_openmp(const Matrix& Q, const Matrix& K, const Matrix& V);

// OpenMP multi-head attention
Matrix multi_head_attention_openmp(const Matrix& Q, const Matrix& K, const Matrix& V, int num_heads);

// OpenMP masked attention (for causal/decoder attention)
Matrix masked_attention_openmp(const Matrix& Q, const Matrix& K, const Matrix& V, bool causal_mask = false);

// OpenMP utility functions
Matrix transpose_openmp(const Matrix& mat);
Matrix multiply_openmp(const Matrix& A, const Matrix& B);
Matrix softmax_openmp(const Matrix& mat);
Matrix scale_openmp(const Matrix& mat, float scalar);

// OpenMP info functions
bool openmp_available();
int get_openmp_max_threads();
void print_openmp_info();
void set_openmp_threads(int num_threads);

#endif // ATTENTION_OPENMP_HPP