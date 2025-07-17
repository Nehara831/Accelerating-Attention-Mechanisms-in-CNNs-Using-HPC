//attention_cuda.cu
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include "attention.hpp"
#include "attention_cuda.hpp"



float* flatten(const Matrix& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    float* flat = new float[rows * cols];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat[i * cols + j] = mat[i][j];
    return flat;
}

Matrix unflatten(const float* flat, int rows, int cols) {
    Matrix mat(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = flat[i * cols + j];
    return mat;
}

//  CUDA Kernels 
__global__ void dot_product_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {

            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void softmax_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float max_val = input[row * N];
        for (int i = 1; i < N; i++) {
            max_val = max(max_val, input[row * N + i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            output[row * N + i] = exp(input[row * N + i] - max_val);
            sum += output[row * N + i];
        }
        
        for (int i = 0; i < N; i++) {
            output[row * N + i] /= sum;
        }
    }
}

//  CUDA Attention Implementation 
Matrix attention_cuda(const Matrix& Q, const Matrix& K, const Matrix& V) {
    
    int M = Q.size();      // Number of queries
    int N = K.size();      // Number of keys
    int K_dim = Q[0].size(); // Dim  queries/keys
    int V_dim = V[0].size(); // Dim values



    // Check matrix dimensions
    if (Q[0].size() != K[0].size()) {
        fprintf(stderr, "Error: Query dimension (%zu) must match key dimension (%zu)\n", Q[0].size(), K[0].size());
        throw std::runtime_error("Query and key dimensions do not match");
    }
    if (K.size() != V.size()) {
        fprintf(stderr, "Error: Number of keys (%zu) must match number of values (%zu)\n", K.size(), V.size());
        throw std::runtime_error("Number of keys and values do not match");
    }


    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_K_T, *d_scores, *d_probs, *d_result;
    cudaMalloc(&d_Q, M * K_dim * sizeof(float));
    cudaMalloc(&d_K, N * K_dim * sizeof(float));
    cudaMalloc(&d_V, N * V_dim * sizeof(float));
    cudaMalloc(&d_K_T, K_dim * N * sizeof(float));
    cudaMalloc(&d_scores, M * N * sizeof(float));
    cudaMalloc(&d_probs, M * N * sizeof(float));
    cudaMalloc(&d_result, M * V_dim * sizeof(float));



    float* h_Q = flatten(Q);
    float* h_K = flatten(K);
    float* h_V = flatten(V);



    cudaMemcpy(d_Q, h_Q, M * K_dim * sizeof(float), cudaMemcpyHostToDevic);
    cudaMemcpy(d_K, h_K, N * K_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N * V_dim * sizeof(float), cudaMemcpyHostToDevice);

    fflush(stderr);

    //  K^T
    float* h_K_T = new float[K_dim * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K_dim; j++) {
            h_K_T[j * N + i] = h_K[i * K_dim + j];
        }
    }
    cudaMemcpy(d_K_T, h_K_T, K_dim * N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_K_T;



    dim3 blockDim(16, 16);
    dim3 gridDimQK((N + blockDim.x - 1) / blockDim.x,
                   (M + blockDim.y - 1) / blockDim.y);
    dim3 gridDimPV((V_dim + blockDim.x - 1) / blockDim.x,
                   (M + blockDim.y - 1) / blockDim.y);

    //  Q * K^T
    dot_product_kernel<<<gridDimQK, blockDim>>>(d_Q, d_K_T, d_scores, M, N, K_dim);
    cudaDeviceSynchronize();
    cudaGetLastError();



    // Softmax
    softmax_kernel<<<gridDimQK, blockDim>>>(d_scores, d_probs, M, N);
    cudaDeviceSynchronize();



    // probs * V
    dot_product_kernel<<<gridDimPV, blockDim>>>(d_probs, d_V, d_result, M, V_dim, N);
    cudaDeviceSynchronize();
    cudaGetLastError();



    float* h_result = new float[M * V_dim];
    cudaMemcpy(h_result, d_result, M * V_dim * sizeof(float), cudaMemcpyDeviceToHost);



    Matrix result = unflatten(h_result, M, V_dim);



    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_K_T);
    cudaFree(d_scores);
    cudaFree(d_probs);
    cudaFree(d_result);

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_result;


    return result;
}

// CUDA utility functions
bool cuda_available() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

int get_cuda_device_count() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess) ? device_count : 0;
}

void print_cuda_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    printf("CUDA Devices: %d\n", device_count);
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    }
}
