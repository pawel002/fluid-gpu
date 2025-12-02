#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm>

// kernel settings
namespace burgers_solver_config {
    constexpr int THREADS = 32;
    constexpr int SMEM_DIM = THREADS + 2;
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

__device__ __forceinline__
float2 load_global(
    const float* __restrict__ u,
    const float* __restrict__ v,
    int x,
    int y,
    int nx,
    int ny)
{
    if (x >= 0 && x < nx && y >= 0 && y < ny) {
        int idx = y * nx + x;
        return make_float2(u[idx], v[idx]);
    }
    return make_float2(0.0f, 0.0f);
}

__global__
void burgers_step(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ u_new,
    float* __restrict__ v_new,
    int nx,
    int ny,
    float cx,
    float cy,
    float diff_coef,
    float inv_dx2,
    float inv_dy2)
{
    // shared memory for the tile + 1-pixel 
    __shared__ float s_u[burgers_solver_config::SMEM_DIM][burgers_solver_config::SMEM_DIM];
    __shared__ float s_v[burgers_solver_config::SMEM_DIM][burgers_solver_config::SMEM_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    // indices within shared memory (shifted by 1 to boundary shift)
    int sx = tx + 1;
    int sy = ty + 1;

    // load center pixel
    float2 center = load_global(u, v, gx, gy, nx, ny);
    s_u[sy][sx] = center.x;
    s_v[sy][sx] = center.y;

    // load shared memory boundaries
    if (tx == 0) {
        float2 val = load_global(u, v, gx - 1, gy, nx, ny);
        s_u[sy][0] = val.x;
        s_v[sy][0] = val.y;
    }
    if (tx == blockDim.x - 1) {
        float2 val = load_global(u, v, gx + 1, gy, nx, ny);
        s_u[sy][sx + 1] = val.x;
        s_v[sy][sx + 1] = val.y;
    }
    if (ty == 0) {
        float2 val = load_global(u, v, gx, gy - 1, nx, ny);
        s_u[0][sx] = val.x;
        s_v[0][sx] = val.y;
    }
    if (ty == blockDim.y - 1) {
        float2 val = load_global(u, v, gx, gy + 1, nx, ny);
        s_u[sy + 1][sx] = val.x;
        s_v[sy + 1][sx] = val.y;
    }

    __syncthreads();

    // early exit for threads completely outside the domain
    if (gx >= nx || gy >= ny) return;

    // boundary check (rim is set to 0)
    if (gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1) {

        float u_C = s_u[sy][sx];
        float u_L = s_u[sy][sx - 1];
        float u_R = s_u[sy][sx + 1];
        float u_T = s_u[sy - 1][sx];
        float u_B = s_u[sy + 1][sx];

        float v_C = s_v[sy][sx];
        float v_L = s_v[sy][sx - 1];
        float v_R = s_v[sy][sx + 1];
        float v_T = s_v[sy - 1][sx];
        float v_B = s_v[sy + 1][sx];

        // 1. horizontal Fluxes (F) for U-equation
        float max_u_R = fmaxf(fabsf(u_C), fabsf(u_R));
        float flux_F_R_u = 0.5f * (0.5f * (u_C * u_C + u_R * u_R)) - 0.5f * max_u_R * (u_R - u_C);

        float max_u_L = fmaxf(fabsf(u_L), fabsf(u_C));
        float flux_F_L_u = 0.5f * (0.5f * (u_L * u_L + u_C * u_C)) - 0.5f * max_u_L * (u_C - u_L);

        // 2. vertical Fluxes (G) for U-equation
        float max_v_B = fmaxf(fabsf(v_C), fabsf(v_B));
        float flux_G_B_u = 0.5f * (v_C * u_C + v_B * u_B) - 0.5f * max_v_B * (u_B - u_C);

        float max_v_T = fmaxf(fabsf(v_T), fabsf(v_C));
        float flux_G_T_u = 0.5f * (v_T * u_T + v_C * u_C) - 0.5f * max_v_T * (u_C - u_T);

        // 3. horizontal Fluxes (F) for V-equation
        float flux_F_R_v = 0.5f * (u_C * v_C + u_R * v_R) - 0.5f * max_u_R * (v_R - v_C);
        float flux_F_L_v = 0.5f * (u_L * v_L + u_C * v_C) - 0.5f * max_u_L * (v_C - v_L);

        // 4. vertical Fluxes (G) for V-equation
        float flux_G_B_v = 0.5f * (0.5f * (v_C * v_C + v_B * v_B)) - 0.5f * max_v_B * (v_B - v_C);
        float flux_G_T_v = 0.5f * (0.5f * (v_T * v_T + v_C * v_C)) - 0.5f * max_v_T * (v_C - v_T);

        // 5. diffusion (5 point laplace)
        float lap_u = (u_R - 2.0f * u_C + u_L) * inv_dx2 + (u_B - 2.0f * u_C + u_T) * inv_dy2;
        float lap_v = (v_R - 2.0f * v_C + v_L) * inv_dx2 + (v_B - 2.0f * v_C + v_T) * inv_dy2;

        // 6. update
        float u_val = u_C - cx * (flux_F_R_u - flux_F_L_u) 
                          - cy * (flux_G_B_u - flux_G_T_u) 
                          + diff_coef * lap_u;

        float v_val = v_C - cx * (flux_F_R_v - flux_F_L_v) 
                          - cy * (flux_G_B_v - flux_G_T_v) 
                          + diff_coef * lap_v;

        int idx = gy * nx + gx;
        u_new[idx] = u_val;
        v_new[idx] = v_val;
    }
    else {
        int idx = gy * nx + gx;
        u_new[idx] = 0.0f;
        v_new[idx] = 0.0f;
    }
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
    int status = 0;

    dim3 blocks(
        (nx + burgers_solver_config::THREADS - 1) / burgers_solver_config::THREADS, 
        (ny + burgers_solver_config::THREADS - 1) / burgers_solver_config::THREADS
    );

    size_t bytes = (size_t) (nx * ny * sizeof(float));

    float dx2       = dx * dx; 
    float dy2       = dy * dy;
    float inv_dx2   = 1.0f / dx2;
    float inv_dy2   = 1.0f / dy2;
    float cx        = dt / dx;
    float cy        = dt / dy;
    float diff_coef = nu * dt;

    CUDA_CHECK(cudaMalloc(&d_u,     bytes));
    CUDA_CHECK(cudaMalloc(&d_v,     bytes));
    CUDA_CHECK(cudaMalloc(&d_u_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_v_new, bytes));

    CUDA_CHECK(cudaMemcpy(d_u, h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));

    // loop
    for (int k = 0; k < steps; k++) {
        burgers_step<<<blocks, burgers_solver_config::threads>>>(
            d_u, d_v, d_u_new, d_v_new,
            nx, ny,
            cx, cy, diff_coef, inv_dx2, inv_dy2);
            
        // swap pointers
        std::swap(d_u, d_u_new);
        std::swap(d_v, d_v_new);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_u, d_u, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost));

cleanup:
    if (d_u) cudaFree(d_u);
    if (d_v) cudaFree(d_v);
    if (d_u_new) cudaFree(d_u_new);
    if (d_v_new) cudaFree(d_v_new);
    return status;
}
