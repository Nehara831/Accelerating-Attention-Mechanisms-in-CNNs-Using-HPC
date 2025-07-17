#ifndef ATTENTION_OPENMP_HPP
#define ATTENTION_OPENMP_HPP

#include "attention.hpp"

Matrix attention_openmp(const Matrix& Q, const Matrix& K, const Matrix& V);


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

#endif 