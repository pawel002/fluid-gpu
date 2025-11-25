#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

// kernel settings
namespace vector_add_config {
    constexpr int THREADS = 512;
}

// error check makro
#define CUDA_CHECK(call)             \
    do {                             \
        cudaError_t err__ = (call);  \
        if (err__ != cudaSuccess)    \
        {                            \
            status = (int)err__;     \
            goto cleanup;            \
        }                            \
    } while (0)


__global__ 
void vector_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}


inline int vector_add_cuda(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int n)
{
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t bytes = (size_t)n * sizeof(float);
    int status = 0;
    int blocks = (n + vector_add_config::THREADS - 1) / vector_add_config::THREADS;

    // allocate device pointers
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // copy inputs
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // kernel
    vector_add_kernel<<<blocks, vector_add_config::THREADS>>>(d_A, d_B, d_C, n);

    // check for errors and sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

cleanup:
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    return status;
}