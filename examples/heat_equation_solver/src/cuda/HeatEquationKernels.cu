#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <nvToolsExt.h>
#include "HeatEquationSolver.h"

__global__ void heat_equation_kernel(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Thread index in X-dimension
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Thread index in Y-dimension

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) //Boundry condition 
    {
        int idx = i * ny + j; 
        float laplacian = (u[idx - ny] + u[idx + ny] - 2 * u[idx]) / (dx * dx) +
                          (u[idx - 1] + u[idx + 1] - 2 * u[idx]) / (dy * dy);
        u_next[idx] = u[idx] + alpha * dt * laplacian;
    }
}

__global__ void heat_equation_kernel_shared_memory(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    extern __shared__ float s_u[];  // Dynamically allocated shared memory

    int i = blockIdx.x * blockDim.x + threadIdx.x; // Thread index in X-dimension
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Thread index in X-dimension

    int tx = threadIdx.x + 1;  // index for shared memory border
    int ty = threadIdx.y + 1;  // index for shared memory border

    int s_idx = tx * (blockDim.y + 2) + ty;  // 2D indexing in shared memory

    // Load into shared memory
    s_u[s_idx] = u[i * ny + j];

    //Grab the appropriate threadsids, accmulate for the warp.
    if (tx == 1 && i > 0) {
        s_u[s_idx - (blockDim.y + 2)] = u[(i - 1) * ny + j];
    }
    if (tx == blockDim.x && i < nx - 1) {
        s_u[s_idx + (blockDim.y + 2)] = u[(i + 1) * ny + j];
    }
    if (ty == 1 && j > 0) {
        s_u[s_idx - 1] = u[i * ny + j - 1];
    }
    if (ty == blockDim.y && j < ny - 1) {
        s_u[s_idx + 1] = u[i * ny + j + 1];
    }
    //To collect the warp and execute in the next cell.
    __syncthreads();


    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) //Boundry condition for threads.
    {
        float laplacian = (s_u[s_idx - (blockDim.y + 2)] + s_u[s_idx + (blockDim.y + 2)] - 2 * s_u[s_idx]) / (dy * dy) +
                          (s_u[s_idx - 1] + s_u[s_idx + 1] - 2 * s_u[s_idx]) / (dx * dx);
        u_next[i * ny + j] = u[i * ny + j] + alpha * dt * laplacian;
    }
}

__global__ void heat_equation_kernel_loop_unroll(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int idx = i * ny + j;
        float laplacian = (u[idx - ny] + u[idx + ny] - 2 * u[idx]) / (dx * dx) +
                          (u[idx - 1] + u[idx + 1] - 2 * u[idx]) / (dy * dy);
        u_next[idx] = u[idx] + alpha * dt * laplacian;
    }
}

extern "C" __host__ void heat_equation_step_gpu(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, KernelType kernelType, dim3 block_size) {
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);

    switch (kernelType) {
        case KernelType::BASIC:
            heat_equation_kernel<<<grid_size, block_size>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        case KernelType::SHARED_MEMORY:
            heat_equation_kernel_shared_memory<<<grid_size, block_size, (block_size.x + 2) * (block_size.y + 2) * sizeof(float)>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        case KernelType::LOOP_UNROLL:
            heat_equation_kernel_loop_unroll<<<grid_size, block_size>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        default:
            // Default
            heat_equation_kernel<<<grid_size, block_size>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
