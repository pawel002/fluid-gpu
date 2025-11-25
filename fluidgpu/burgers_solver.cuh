#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

// kernel settings
namespace burgers_solver_config {
    constexpr int THREADS = 32;
    constexpr dim3 threads = dim3(THREADS, THREADS);
}

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
void burgers_step(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ u_new,
    float* __restrict__ v_new,
    int nx,
    int ny,
    float nu,
    float dt,
    float dx,
    float dy,
    size_t pitch_bytes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    const float* row_u     = (const float*)((const char*)u + j * pitch_bytes);
    const float* row_u_new = (float*)((char*)u_new + j * pitch_bytes);

    float val = row_u[i];

    // TODO: implement
}


inline int compute_burgers_steps(
    float* h_u,
    float* h_v,
    const int nx,
    const int ny,
    const float nu,
    const float dt,
    const float dx,
    const float dy,
    int steps)
{
    float *d_u = nullptr, *d_v = nullptr, *d_u_new = nullptr, *d_v_new = nullptr;
    size_t pitch = 0;
    int status = 0;

    dim3 blocks(
        (nx + burgers_solver_config::THREADS - 1) / burgers_solver_config::THREADS, 
        (ny + burgers_solver_config::THREADS - 1) / burgers_solver_config::THREADS
    );

    // allocate device pointers
    CUDA_CHECK(cudaMallocPitch(&d_u,     &pitch, nx * sizeof(float), ny));
    CUDA_CHECK(cudaMallocPitch(&d_v,     &pitch, nx * sizeof(float), ny));
    CUDA_CHECK(cudaMallocPitch(&d_u_new, &pitch, nx * sizeof(float), ny));
    CUDA_CHECK(cudaMallocPitch(&d_v_new, &pitch, nx * sizeof(float), ny));

    // copy inputs - both for u,v, u_new and v_new
    CUDA_CHECK(cudaMemcpy2D(d_u,     pitch, h_u, nx * sizeof(float), nx * sizeof(float), ny, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_u_new, pitch, h_u, nx * sizeof(float), nx * sizeof(float), ny, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_v,     pitch, h_v, nx * sizeof(float), nx * sizeof(float), ny, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_v_new, pitch, h_v, nx * sizeof(float), nx * sizeof(float), ny, cudaMemcpyHostToDevice));

    // loop
    float *temp_u, *temp_v;
    for (int k = 0; k < steps; k++) {
        burgers_step<<<blocks, burgers_solver_config::threads>>>(
            d_u, d_v, d_u_new, d_v_new, nx, ny, nu, dt, dx, dy, pitch);
            
        // Swap pointers
        temp_u = d_u; d_u = d_u_new; d_u_new = temp_u;
        temp_v = d_v; d_v = d_v_new; d_v_new = temp_v;
    }

    // sync + errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back to host
    CUDA_CHECK(cudaMemcpy2D(h_u, nx * sizeof(float), d_u, pitch, nx * sizeof(float), ny, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_v, nx * sizeof(float), d_v, pitch, nx * sizeof(float), ny, cudaMemcpyDeviceToHost));

cleanup:
    if (d_u) cudaFree(d_u);
    if (d_v) cudaFree(d_v);
    if (d_u_new) cudaFree(d_u_new);
    if (d_v_new) cudaFree(d_v_new);
    return status;
}