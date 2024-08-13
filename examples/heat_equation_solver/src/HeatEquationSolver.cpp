#include "HeatEquationSolver.h"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

HeatEquationSolver::HeatEquationSolver(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads)
    : nx_(nx), ny_(ny), dx_(dx), dy_(dy), dt_(dt), alpha_(alpha), num_threads_(num_threads) {
#ifdef GGML_USE_CUDA
    nvtxMarkA("HeatEquationSolver Constructor");
#endif
    size_t ctx_size = nx_ * ny_ * sizeof(float) * 2 + 1024;
    struct ggml_init_params params = {ctx_size, NULL, false};
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to initialize ggml context");
    }

    temp_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, nx, ny);
    temp_next_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, nx, ny);

#ifdef GGML_USE_CUDA
    nvtxRangePush("CUDA Memory Allocation");
    CUDA_CHECK(cudaMalloc(&d_temp_, nx_ * ny_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_next_, nx_ * ny_ * sizeof(float)));
    nvtxRangePop();
#endif
}

HeatEquationSolver::~HeatEquationSolver() {
#ifdef GGML_USE_CUDA
    nvtxMarkA("HeatEquationSolver Destructor");
#endif
    ggml_free(ctx_);

#ifdef GGML_USE_CUDA
    CUDA_CHECK(cudaFree(d_temp_));
    CUDA_CHECK(cudaFree(d_temp_next_));
#endif
}

void HeatEquationSolver::set_initial_condition(const std::vector<float>& initial_temp) {
#ifdef GGML_USE_CUDA
    nvtxMarkA("Set Initial Condition");
#endif
    if (initial_temp.size() != static_cast<size_t>(nx_ * ny_)) {
        throw std::invalid_argument("Initial temperature vector size does not match grid size");
    }
    memcpy(temp_->data, initial_temp.data(), nx_ * ny_ * sizeof(float));

#ifdef GGML_USE_CUDA
    CUDA_CHECK(cudaMemcpy(d_temp_, initial_temp.data(), nx_ * ny_ * sizeof(float), cudaMemcpyHostToDevice));
#endif
}

void HeatEquationSolver::solve_cpu(int num_steps) {
#ifdef GGML_USE_CUDA
    nvtxRangePush("CPU Solve");
#endif

    float* u = (float*)temp_->data;
    float* u_next = (float*)temp_next_->data;

#ifdef USE_OPENMP
    omp_set_num_threads(num_threads_);
    std::cout << "Using OpenMP with " << num_threads_ << " threads for CPU computation." << std::endl;
#endif

    for (int step = 0; step < num_steps; ++step) {
#ifdef GGML_USE_CUDA
        nvtxRangePush("CPU Step");
#endif
#ifdef USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int i = 1; i < nx_ - 1; ++i) {
            for (int j = 1; j < ny_ - 1; ++j) {
                int idx = i * ny_ + j;
                float laplacian = (u[idx - ny_] + u[idx + ny_] - 2 * u[idx]) / (dx_ * dx_) +
                                  (u[idx - 1] + u[idx + 1] - 2 * u[idx]) / (dy_ * dy_);
                u_next[idx] = u[idx] + alpha_ * dt_ * laplacian;
            }
        }

        // Apply boundary conditions (assuming Dirichlet conditions)
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < nx_; ++i) {
            u_next[i * ny_] = u[i * ny_];
            u_next[i * ny_ + ny_ - 1] = u[i * ny_ + ny_ - 1];
        }
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < ny_; ++j) {
            u_next[j] = u[j];
            u_next[(nx_ - 1) * ny_ + j] = u[(nx_ - 1) * ny_ + j];
        }

        std::swap(u, u_next);
        std::swap(temp_, temp_next_);

        if (step % 1000 == 0) {
            std::cout << "CPU Step " << step << ", Center temp: " << u[nx_/2 * ny_ + ny_/2] << std::endl;
        }
#ifdef GGML_USE_CUDA
        nvtxRangePop();
#endif
    }
#ifdef GGML_USE_CUDA
    nvtxRangePop();
#endif
}

void HeatEquationSolver::solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) {
#ifdef GGML_USE_CUDA
    nvtxRangePush("GPU Solve");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 grid_size((nx_ + block_size.x - 1) / block_size.x, (ny_ + block_size.y - 1) / block_size.y);

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all previous operations are done

    for (int step = 0; step < num_steps; ++step) {
        nvtxRangePush("GPU Step");

        // Pass block_size to heat_equation_step_gpu
        heat_equation_step_gpu(d_temp_, d_temp_next_, nx_, ny_, alpha_, dx_, dy_, dt_, kernelType, block_size);

        CUDA_CHECK(cudaDeviceSynchronize()); // Synchronize after kernel execution
        nvtxRangePop(); // End of GPU Step marker

        std::swap(d_temp_, d_temp_next_);

        // if (step % 1000 == 0) {
        //     nvtxRangePush("GPU Memory Copy to Host");
        //     float center_temp;
        //     CUDA_CHECK(cudaMemcpy(&center_temp, d_temp_ + nx_/2 * ny_ + ny_/2, sizeof(float), cudaMemcpyDeviceToHost));
        //     CUDA_CHECK(cudaDeviceSynchronize());
        //     // std::cout << "GPU Step " << step << ", Center temp: " << center_temp << std::endl;
        //     nvtxRangePop();
        // }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    nvtxRangePush("Final GPU Memory Copy to Host");
    CUDA_CHECK(cudaMemcpy(temp_->data, d_temp_, nx_ * ny_ * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();

    nvtxRangePop(); // End of GPU Solve marker
#else
    std::cerr << "CUDA not available. Using CPU solver instead." << std::endl;
    solve_cpu(num_steps);
#endif
}

std::vector<float> HeatEquationSolver::get_temperature_field() const {
#ifdef GGML_USE_CUDA
    nvtxRangePush("Get Temperature Field");
#endif
    std::vector<float> result(nx_ * ny_);
    memcpy(result.data(), temp_->data, nx_ * ny_ * sizeof(float));
#ifdef GGML_USE_CUDA
    nvtxRangePop();
#endif
    return result;
}

void HeatEquationSolver::swap_tensors() {
    std::swap(temp_, temp_next_);
}
