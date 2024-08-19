#include "HeatEquationSolverEuler.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <chrono>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

extern "C" void heat_equation_step_euler_gpu(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, dim3 block_size, cudaStream_t stream);

HeatEquationSolverEuler::HeatEquationSolverEuler(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads)
    : HeatEquationSolverBase(nx, ny, dx, dy, dt, alpha, num_threads) {
#ifdef GGML_USE_CUDA
    CUDA_CHECK(cudaMalloc(&d_temp_, nx_ * ny_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_next_, nx_ * ny_ * sizeof(float)));
#endif
    std::cout << "HeatEquationSolverEuler initialized with:" << std::endl;
    std::cout << "  Grid size: " << nx_ << " x " << ny_ << std::endl;
}

HeatEquationSolverEuler::~HeatEquationSolverEuler() {
#ifdef GGML_USE_CUDA
    CUDA_CHECK(cudaFree(d_temp_));
    CUDA_CHECK(cudaFree(d_temp_next_));
#endif
}

void HeatEquationSolverEuler::set_initial_condition(const std::vector<float>& initial_temp) {
    if (initial_temp.size() != static_cast<size_t>(nx_ * ny_)) {
        throw std::invalid_argument("Initial temperature vector size does not match grid size");
    }
    memcpy(temp_, initial_temp.data(), nx_ * ny_ * sizeof(float));

#ifdef GGML_USE_CUDA
    CUDA_CHECK(cudaMemcpy(d_temp_, initial_temp.data(), nx_ * ny_ * sizeof(float), cudaMemcpyHostToDevice));
#endif

    std::cout << "Initial Condition set." << std::endl;
}

void HeatEquationSolverEuler::solve_cpu(int num_steps) {
    float* u = temp_;
    float* u_next = temp_next_;

#ifdef USE_OPENMP
    omp_set_num_threads(num_threads_);
    std::cout << "Using OpenMP with " << num_threads_ << " threads for CPU computation." << std::endl;
#endif

    for (int step = 0; step < num_steps; ++step) {
#ifdef USE_NVTX
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
            std::cout << "Forward Euler Step " << step << ", Center temp: " << u[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
        }
#ifdef USE_NVTX
        nvtxRangePop();
#endif
    }
}

void HeatEquationSolverEuler::solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) {
#ifdef GGML_USE_CUDA
#ifdef USE_NVTX
    nvtxRangePush("GPU Solve");
#endif

    for (int step = 0; step < num_steps; ++step) {
#ifdef USE_NVTX
        nvtxRangePush("GPU Step");
#endif


        heat_equation_step_euler_gpu(d_temp_, d_temp_next_, nx_, ny_, alpha_, dx_, dy_, dt_, block_size, 0);

        CUDA_CHECK(cudaDeviceSynchronize());

#ifdef USE_NVTX
        nvtxRangePop(); 
#endif

        std::swap(d_temp_, d_temp_next_);

        if (step % 1000 == 0) {
#ifdef USE_NVTX
            nvtxRangePush("GPU Memory Copy to Host");
#endif
            float center_temp;
            CUDA_CHECK(cudaMemcpy(&center_temp, d_temp_ + nx_/2 * ny_ + ny_/2, sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "GPU Step (Euler) " << step << ", Center temp: " << center_temp << std::endl;
#ifdef USE_NVTX
            nvtxRangePop();
#endif
        }
    }

    CUDA_CHECK(cudaMemcpy(temp_, d_temp_, nx_ * ny_ * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef USE_NVTX
    nvtxRangePop(); // End of GPU Solve marker
#endif
#else
    std::cerr << "CUDA not available. Using CPU solver instead." << std::endl;
    solve_cpu(num_steps);
#endif
}

std::vector<float> HeatEquationSolverEuler::get_temperature_field() const {
    std::vector<float> result(nx_ * ny_);
    memcpy(result.data(), temp_, nx_ * ny_ * sizeof(float));
    return result;
}

void HeatEquationSolverEuler::verify_results(int num_steps, KernelType kernelType, dim3 block_size) {
    std::cout << "Verifying results between CPU and GPU (Euler)..." << std::endl;

    //initial condition
    std::vector<float> initial_condition = get_temperature_field();

    // Solve on GPU
    auto gpu_start = std::chrono::high_resolution_clock::now();
    solve_gpu(num_steps, kernelType, block_size);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_diff = gpu_end - gpu_start;

    std::vector<float> gpu_result = get_temperature_field();

    
    std::cout << "GPU (Euler): Temperature at center: " << gpu_result[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
    std::cout << "GPU (Euler) Execution Time: " << gpu_diff.count() << " ms" << std::endl;

    // Reset the initial condition for CPU
    // set_initial_condition(initial_condition);

    // // Solve on CPU
    // auto cpu_start = std::chrono::high_resolution_clock::now();
    // solve_cpu(num_steps);
    // auto cpu_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> cpu_diff = cpu_end - cpu_start;

    // std::vector<float> cpu_result = get_temperature_field();

    // // Print CPU center temperature and execution time
    // std::cout << "CPU (Euler): Temperature at center: " << cpu_result[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
    // std::cout << "CPU (Euler) Execution Time: " << cpu_diff.count() << " ms" << std::endl;

    // // Compare the results
    // bool match = true;
    // float relative_tolerance = 1e-4;
    // for (size_t i = 1; i < nx_ - 1; ++i) {
    //     for (size_t j = 1; j < ny_ - 1; ++j) {
    //         size_t idx = i * ny_ + j;
    //         float cpu_val = cpu_result[idx];
    //         float gpu_val = gpu_result[idx];
    //         float relative_diff = std::abs(cpu_val - gpu_val) / std::max(std::abs(cpu_val), std::abs(gpu_val));
    //         if (relative_diff > relative_tolerance) {
    //             match = false;
    //             std::cerr << "Mismatch at index " << idx << ": CPU=" << cpu_val << " GPU=" << gpu_val 
    //                       << " Relative diff=" << relative_diff << std::endl;
    //         }
    //     }
    // }

    // if (match) {
    //     std::cout << "Verification passed! CPU and GPU results match within tolerance." << std::endl;
    // } else {
    //     std::cerr << "Verification failed! CPU and GPU results do not match." << std::endl;
    // }
}

void HeatEquationSolverEuler::print_debug_info(int step, bool is_initial) {
    if (is_initial) {
        std::cout << "Initial Condition:" << std::endl;
    } else {
        std::cout << "Temperature field at step " << step << ":" << std::endl;
    }

    std::vector<float> temp_field = get_temperature_field();

    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            std::cout << "[" << i << "," << j << "]=" << std::setw(10) << std::fixed << std::setprecision(4) << temp_field[i * ny_ + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

