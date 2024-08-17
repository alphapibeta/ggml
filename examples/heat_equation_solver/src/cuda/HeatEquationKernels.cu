



#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>

// Include this header to get the KernelType enum definition
#include "../include/HeatEquationSolverBase.h"

// Kernel declarations
__global__ void heat_equation_kernel_basic(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_shared_memory(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_loop_unroll(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_shared_memory_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_loop_unroll_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_two_stream(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, int stream_id);

// Implementation of heat_equation_step_gpu
extern "C" __host__ void heat_equation_step_gpu(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, KernelType kernelType, dim3 block_size, cudaStream_t stream, int stream_id) {
    dim3 grid_size;
    
    if (kernelType == KernelType::TWO_STREAM) {
        // For stream-based kernel, we only process half of the grid in each stream
        grid_size = dim3((nx/2 + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
    } else {
        // For all other cases
        grid_size = dim3((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
    }

    switch (kernelType) {
        case KernelType::BASIC:
            heat_equation_kernel_basic<<<grid_size, block_size, 0, stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        case KernelType::SHARED_MEMORY:
            heat_equation_kernel_shared_memory<<<grid_size, block_size, (block_size.x + 2) * (block_size.y + 2) * sizeof(float), stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        case KernelType::LOOP_UNROLL:
            heat_equation_kernel_loop_unroll<<<grid_size, block_size, 0, stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        case KernelType::FMA:
            heat_equation_kernel_fma<<<grid_size, block_size, 0, stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        case KernelType::SHARED_MEMORY_FMA:
            heat_equation_kernel_shared_memory_fma<<<grid_size, block_size, (block_size.x + 2) * (block_size.y + 2) * sizeof(float), stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        case KernelType::LOOP_UNROLL_FMA:
            heat_equation_kernel_loop_unroll_fma<<<grid_size, block_size, 0, stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
        case KernelType::TWO_STREAM:
            heat_equation_kernel_two_stream<<<grid_size, block_size, 0, stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt, stream_id);
            break;
        default:
            heat_equation_kernel_basic<<<grid_size, block_size, 0, stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt);
            break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}


// Implement your kernel functions here (keep your existing implementations)
__global__ void heat_equation_kernel_basic(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we are within bounds and not on the edges
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        int idx = i * ny + j;
        // Calculate Laplacian
        float laplacian = (u[idx - ny] + u[idx + ny] - 2 * u[idx]) / (dx * dx) +
                          (u[idx - 1] + u[idx + 1] - 2 * u[idx]) / (dy * dy);
        // Update temperature
        u_next[idx] = u[idx] + alpha * dt * laplacian;
    }

    // Apply boundary conditions
    if (i == 0) { // Top boundary
        u_next[i * ny + j] = 1000.0f; // Maintain fixed temperature at top boundary
    } else if (i == nx - 1) { // Bottom boundary
        u_next[i * ny + j] = u[i * ny + j]; // Copy existing temperature
    }

    if (j == 0) { // Left boundary
        u_next[i * ny + j] = u[i * ny + j];
    } else if (j == ny - 1) { // Right boundary
        u_next[i * ny + j] = u[i * ny + j];
    }
}



__global__ void heat_equation_kernel_shared_memory(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    extern __shared__ float s_u[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    int s_idx = tx * (blockDim.y + 2) + ty;

    s_u[s_idx] = u[i * ny + j];

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

    __syncthreads();

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        float laplacian = (s_u[s_idx - (blockDim.y + 2)] + s_u[s_idx + (blockDim.y + 2)] - 2.0f * s_u[s_idx]) / (dx * dx) +
                          (s_u[s_idx - 1] + s_u[s_idx + 1] - 2.0f * s_u[s_idx]) / (dy * dy);
        u_next[i * ny + j] = u[i * ny + j] + alpha * dt * laplacian;
    }
}

__global__ void heat_equation_kernel_loop_unroll(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int idx = i * ny + j;
        float laplacian = __fmaf_rn((u[idx - ny] + u[idx + ny] - 2.0f * u[idx]), 1.0f / (dx * dx), 
                                    (u[idx - 1] + u[idx + 1] - 2.0f * u[idx]) / (dy * dy));
        u_next[idx] = __fmaf_rn(alpha, dt * laplacian, u[idx]);
    }
}

__global__ void heat_equation_kernel_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int idx = i * ny + j;
        float laplacian = __fmaf_rn((u[idx - ny] + u[idx + ny] - 2.0f * u[idx]), 1.0f / (dx * dx), 
                                    (u[idx - 1] + u[idx + 1] - 2.0f * u[idx]) / (dy * dy));
        u_next[idx] = __fmaf_rn(alpha, dt * laplacian, u[idx]);
    }
}

__global__ void heat_equation_kernel_shared_memory_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    extern __shared__ float s_u[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    int s_idx = tx * (blockDim.y + 2) + ty;

    s_u[s_idx] = u[i * ny + j];

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

    __syncthreads();

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        float laplacian = __fmaf_rn((s_u[s_idx - (blockDim.y + 2)] + s_u[s_idx + (blockDim.y + 2)] - 2.0f * s_u[s_idx]), 1.0f / (dx * dx), 
                                    (s_u[s_idx - 1] + s_u[s_idx + 1] - 2.0f * s_u[s_idx]) / (dy * dy));
        u_next[i * ny + j] = __fmaf_rn(alpha, dt * laplacian, u[i * ny + j]);
    }
}

__global__ void heat_equation_kernel_loop_unroll_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int idx = i * ny + j;
        float laplacian = __fmaf_rn((u[idx - ny] + u[idx + ny] - 2.0f * u[idx]), 1.0f / (dx * dx), 
                                    (u[idx - 1] + u[idx + 1] - 2.0f * u[idx]) / (dy * dy));
        u_next[idx] = __fmaf_rn(alpha, dt * laplacian, u[idx]);
    }
}


__global__ void heat_equation_kernel_two_stream(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, int stream_id){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Adjust for stream partitioning and ghost cells
    int partition_nx = nx / 2;  // Number of grid points in x for each partition (excluding ghost cells)
    // int global_i = i + stream_id * partition_nx;  // Compute global i-index// Ensure we are within valid partition boundaries (excluding ghost cells)
    if (i >= 1 && i < partition_nx + 1 && j >= 1 && j < ny - 1) {
        // Calculate index within partition, considering ghost cells
        int idx = i * ny + j;

        // Compute the Laplacian using finite differences
        float laplacian = (u[idx - ny] + u[idx + ny] - 2.0f * u[idx]) / (dx * dx) +
                          (u[idx - 1] + u[idx + 1] - 2.0f * u[idx]) / (dy * dy);

        // Update the temperature
        u_next[idx] = u[idx] + alpha * dt * laplacian;

        // Optional: Debugging output for a key point (e.g., the center of the entire grid)
        // if (stream_id == 0 && i == partition_nx / 2 && j == ny / 2) {
        //     printf("Step center temp: %f at (global_i=%d, j=%d)\n", u_next[idx], global_i, j);
        // }
    }
}
