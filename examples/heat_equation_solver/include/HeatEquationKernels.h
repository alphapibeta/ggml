#ifndef HEAT_EQUATION_KERNELS_H
#define HEAT_EQUATION_KERNELS_H

#include <cuda_runtime.h>
#include "HeatEquationSolverBase.h"

__global__ void heat_equation_kernel_basic(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_shared_memory(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_loop_unroll(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_shared_memory_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_loop_unroll_fma(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt);
__global__ void heat_equation_kernel_two_stream(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, int stream_id);

extern "C" void heat_equation_step_gpu(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, KernelType kernelType, dim3 block_size, cudaStream_t stream, int stream_id = -1);

#endif // HEAT_EQUATION_KERNELS_H
