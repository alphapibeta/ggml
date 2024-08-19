



#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>

// Include this header to get the KernelType enum definition
#include "../include/HeatEquationSolverBase.h"
#include "../include/HeatEquationKernels.h"



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
    } else if (i == 0) {
        u_next[i * ny + j] = 1000.0f;  
    } else if (i == nx - 1 || j == 0 || j == ny - 1) {
        u_next[i * ny + j] = u[i * ny + j];  
    }
}

__global__ void heat_equation_kernel_loop_unroll(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int idx = i * ny + j;
        float laplacian_x = 0.0f;
        float laplacian_y = 0.0f;


        #pragma unroll
        for (int offset = -1; offset <= 1; offset += 2) {
            laplacian_x = __fmaf_rn(u[idx + offset * ny], 1.0f / (dx * dx), laplacian_x);
        }
        laplacian_x = __fmaf_rn(-2.0f * u[idx], 1.0f / (dx * dx), laplacian_x);


        #pragma unroll
        for (int offset = -1; offset <= 1; offset += 2) {
            laplacian_y = __fmaf_rn(u[idx + offset], 1.0f / (dy * dy), laplacian_y);
        }
        laplacian_y = __fmaf_rn(-2.0f * u[idx], 1.0f / (dy * dy), laplacian_y);

        float laplacian = laplacian_x + laplacian_y;
        u_next[idx] = __fmaf_rn(alpha * dt, laplacian, u[idx]);
    } else if (i == 0) {
        u_next[i * ny + j] = 1000.0f; 
    } else if (i == nx - 1 || j == 0 || j == ny - 1) {
        u_next[i * ny + j] = u[i * ny + j]; 
    }
}


__global__ void heat_equation_kernel_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int idx = i * ny + j;
        float laplacian_x = __fmaf_rn(u[idx - ny] + u[idx + ny], 1.0f / (dx * dx), -2.0f * u[idx] / (dx * dx));
        float laplacian_y = __fmaf_rn(u[idx - 1] + u[idx + 1], 1.0f / (dy * dy), -2.0f * u[idx] / (dy * dy));
        float laplacian = laplacian_x + laplacian_y;
        u_next[idx] = __fmaf_rn(alpha * dt, laplacian, u[idx]);
    } else if (i == 0) {
        u_next[i * ny + j] = 1000.0f;  
    } else if (i == nx - 1 || j == 0 || j == ny - 1) {
        u_next[i * ny + j] = u[i * ny + j]; 
    }
}


__global__ void heat_equation_kernel_shared_memory_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
    extern __shared__ float s_u[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    int s_idx = tx * (blockDim.y + 2) + ty;


    if (i < nx && j < ny) {
        s_u[s_idx] = u[i * ny + j];
    }


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
        float laplacian_x = __fmaf_rn(s_u[s_idx - (blockDim.y + 2)] + s_u[s_idx + (blockDim.y + 2)], 1.0f / (dx * dx), -2.0f * s_u[s_idx] / (dx * dx));
        float laplacian_y = __fmaf_rn(s_u[s_idx - 1] + s_u[s_idx + 1], 1.0f / (dy * dy), -2.0f * s_u[s_idx] / (dy * dy));
        float laplacian = laplacian_x + laplacian_y;
        u_next[i * ny + j] = __fmaf_rn(alpha * dt, laplacian, u[i * ny + j]);
    } else if (i == 0) {
        u_next[i * ny + j] = 1000.0f; 
    } else if (i == nx - 1 || j == 0 || j == ny - 1) {
        u_next[i * ny + j] = u[i * ny + j];  
    }
}


__global__ void heat_equation_kernel_loop_unroll_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int idx = i * ny + j;
        float laplacian_x = 0.0f;
        float laplacian_y = 0.0f;


        #pragma unroll
        for (int offset = -1; offset <= 1; offset += 2) {
            laplacian_x = __fmaf_rn(u[idx + offset * ny], 1.0f / (dx * dx), laplacian_x);
        }
        laplacian_x = __fmaf_rn(-2.0f * u[idx], 1.0f / (dx * dx), laplacian_x);


        #pragma unroll
        for (int offset = -1; offset <= 1; offset += 2) {
            laplacian_y = __fmaf_rn(u[idx + offset], 1.0f / (dy * dy), laplacian_y);
        }
        laplacian_y = __fmaf_rn(-2.0f * u[idx], 1.0f / (dy * dy), laplacian_y);

        float laplacian = laplacian_x + laplacian_y;
        u_next[idx] = __fmaf_rn(alpha * dt, laplacian, u[idx]);
    } else if (i == 0) {
        u_next[i * ny + j] = 1000.0f;  
    } else if (i == nx - 1 || j == 0 || j == ny - 1) {
        u_next[i * ny + j] = u[i * ny + j];  
    }
}


__global__ void heat_equation_kernel_two_stream(float* d_temp, float* d_temp_next, int part_nx, int ny, float alpha, float dx, float dy, float dt, int stream_id) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int NUM_STREAMS=2;

    int nx_with_ghost = part_nx + 2; 
    int idx = y * nx_with_ghost + x;

    if (x > 0 && x < nx_with_ghost - 1 && y > 0 && y < ny - 1) {
        float laplacian = (d_temp[idx - nx_with_ghost] + d_temp[idx + nx_with_ghost] - 2 * d_temp[idx]) / (dx * dx) +
                          (d_temp[idx - 1] + d_temp[idx + 1] - 2 * d_temp[idx]) / (dy * dy);
        d_temp_next[idx] = d_temp[idx] + alpha * dt * laplacian;
    }


    if (y == 0) {
        d_temp_next[idx] = 1000.0f; 
    } else if (y == ny - 1 || x == 1 || x == nx_with_ghost - 2) {

        d_temp_next[idx] = d_temp[idx];
    }


    if (x == nx_with_ghost - 2 && stream_id < NUM_STREAMS - 1) {
        d_temp_next[idx + 1] = d_temp_next[idx];
    } else if (x == 1 && stream_id > 0) {
        d_temp_next[idx - 1] = d_temp_next[idx];
    }
}




// RK4 kernels
__global__ void heat_equation_kernel_rk4(float* u, float* k, int nx, int ny, float alpha, float dx, float dy, float dt, int stream_id, float* prev_k, float k_factor) {
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
        

        // if (i == nx/2 && j == ny/2) {
        //     printf("GPU: i=%d, j=%d, u=%f, laplacian=%f, u_next=%f\n", i, j, u[idx], laplacian, u_next[idx]);
        // }
    }


    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        int idx = i * ny + j;
        u_next[idx] = u[idx]; 
    }
}


// Host functions

extern "C" __host__ void heat_equation_step_gpu(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, KernelType kernelType, dim3 block_size, cudaStream_t stream, int stream_id) {
    dim3 grid_size;
    
    if (kernelType == KernelType::TWO_STREAM || kernelType == KernelType::RK4) {
        int partition_nx = (kernelType == KernelType::TWO_STREAM) ? nx / 2 : nx / 4;
        grid_size = dim3((partition_nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
    } else {
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
        case KernelType::RK4:
            // RK4 is handled separately in heat_equation_step_rk4_gpu
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



extern "C" __host__ void heat_equation_step_rk4_gpu(float* u, float* u_next, float* k1, float* k2, float* k3, float* k4, int nx, int ny, float alpha, float dx, float dy, float dt, dim3 block_size, cudaStream_t stream, int stream_id) {
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);

    heat_equation_kernel_rk4<<<grid_size, block_size, 0, stream>>>(u, k1, nx, ny, alpha, dx, dy, dt, stream_id, nullptr, 0.0f);
    heat_equation_kernel_rk4<<<grid_size, block_size, 0, stream>>>(u, k2, nx, ny, alpha, dx, dy, dt, stream_id, k1, dt/2.0f);
    heat_equation_kernel_rk4<<<grid_size, block_size, 0, stream>>>(u, k3, nx, ny, alpha, dx, dy, dt, stream_id, k2, dt/2.0f);
    heat_equation_kernel_rk4<<<grid_size, block_size, 0, stream>>>(u, k4, nx, ny, alpha, dx, dy, dt, stream_id, k3, dt);

    combine_rk4_steps<<<grid_size, block_size, 0, stream>>>(u, u_next, k1, k2, k3, k4, nx, ny, dt);
}


extern "C" __host__ void heat_equation_step_euler_gpu(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, dim3 block_size, cudaStream_t stream) {
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);

    // Euler kernel
    heat_equation_kernel_euler<<<grid_size, block_size, 0, stream>>>(u, u_next, nx, ny, alpha, dx, dy, dt);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in Euler method: %s\n", cudaGetErrorString(err));
    }
}



__host__ void heat_equation_step_rk4_gpu_wrapper(float* u, float* u_next, float* k1, float* k2, float* k3, float* k4,
    int nx, int ny, float alpha, float dx, float dy, float dt,
    dim3 grid_size, dim3 block_size, cudaStream_t stream, int stream_id) {
    float dt_half = dt / 2.0f;
    float dt_sixth = dt / 6.0f;

    heat_equation_kernel_rk4<<<grid_size, block_size, 0, stream>>>(u, k1, nx, ny, alpha, dx, dy, dt, stream_id, nullptr, 0.0f);
    heat_equation_kernel_rk4<<<grid_size, block_size, 0, stream>>>(u, k2, nx, ny, alpha, dx, dy, dt_half, stream_id, k1, 0.5f);
    heat_equation_kernel_rk4<<<grid_size, block_size, 0, stream>>>(u, k3, nx, ny, alpha, dx, dy, dt_half, stream_id, k2, 0.5f);
    heat_equation_kernel_rk4<<<grid_size, block_size, 0, stream>>>(u, k4, nx, ny, alpha, dx, dy, dt, stream_id, k3, 1.0f);

    combine_rk4_steps<<<grid_size, block_size, 0, stream>>>(u, u_next, k1, k2, k3, k4, nx, ny, dt_sixth);
}