#ifndef ATTENTION_CUDA_HPP
#define ATTENTION_CUDA_HPP

#include "attention.hpp"

// CUDA attention implementation
Matrix attention_cuda(const Matrix& Q, const Matrix& K, const Matrix& V);

// CUDA utilities
bool cuda_available();
int get_cuda_device_count();
void print_cuda_info();

#endif