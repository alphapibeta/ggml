#include "HeatEquationSolverNoStreams.h"
#include "HeatEquationKernels.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <nvToolsExt.h>

#ifdef USE_OPENMP
#include <omp.h>
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

HeatEquationSolverNoStreams::HeatEquationSolverNoStreams(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads)
    : HeatEquationSolverBase(nx, ny, dx, dy, dt, alpha, num_threads) {
#ifdef GGML_USE_CUDA
    nvtxMarkA("HeatEquationSolverNoStreams Constructor");

    CUDA_CHECK(cudaMalloc(&d_temp_, nx_ * ny_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_next_, nx_ * ny_ * sizeof(float)));
#endif
    std::cout << "HeatEquationSolverNoStreams initialized with:" << std::endl;
    std::cout << "  Grid size: " << nx_ << " x " << ny_ << std::endl;
}

HeatEquationSolverNoStreams::~HeatEquationSolverNoStreams() {
#ifdef GGML_USE_CUDA
    CUDA_CHECK(cudaFree(d_temp_));
    CUDA_CHECK(cudaFree(d_temp_next_));
#endif
}

void HeatEquationSolverNoStreams::set_initial_condition(const std::vector<float>& initial_temp) {
    if (initial_temp.size() != static_cast<size_t>(nx_ * ny_)) {
        throw std::invalid_argument("Initial temperature vector size does not match grid size");
    }
    memcpy(temp_, initial_temp.data(), nx_ * ny_ * sizeof(float));

#ifdef GGML_USE_CUDA
    CUDA_CHECK(cudaMemcpy(d_temp_, initial_temp.data(), nx_ * ny_ * sizeof(float), cudaMemcpyHostToDevice));
#endif

    std::cout << "Initial Condition set." << std::endl;
}

void HeatEquationSolverNoStreams::solve_cpu(int num_steps) {
    float* u = temp_;
    float* u_next = temp_next_;

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
}


void HeatEquationSolverNoStreams::verify_results(int num_steps, KernelType kernelType, dim3 block_size) {
    std::cout << "Verifying results between CPU and GPU without streams..." << std::endl;

    // Store the initial condition
    std::vector<float> initial_condition = get_temperature_field();

    // Solve on GPU
    auto gpu_start = std::chrono::high_resolution_clock::now();
    solve_gpu(num_steps, kernelType, block_size);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_diff = gpu_end - gpu_start;

    std::vector<float> gpu_result = get_temperature_field();

    // Print GPU center temperature and execution time
    std::cout << "GPU: Temperature at center: " << gpu_result[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
    std::cout << "GPU Execution Time: " << gpu_diff.count() << " ms" << std::endl;

    // Reset the initial condition for CPU
    set_initial_condition(initial_condition);

    // Solve on CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    solve_cpu(num_steps);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_diff = cpu_end - cpu_start;

    std::vector<float> cpu_result = get_temperature_field();

    // Print CPU center temperature and execution time
    std::cout << "CPU: Temperature at center: " << cpu_result[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
    std::cout << "CPU Execution Time: " << cpu_diff.count() << " ms" << std::endl;

    // Compare results
    const float relative_tolerance = 1e-5f;
    const float absolute_tolerance = 1e-8f;
    int mismatch_count = 0;
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        float abs_diff = std::abs(cpu_result[i] - gpu_result[i]);
        float abs_max = std::max(std::abs(cpu_result[i]), std::abs(gpu_result[i]));
        if (abs_diff > relative_tolerance * abs_max + absolute_tolerance) {
            if (mismatch_count < 10) {  // Print details for the first 10 mismatches
                int x = i / ny_;
                int y = i % ny_;
                std::cerr << "Mismatch at index " << i << " (x=" << x << ", y=" << y << "): CPU value = " << cpu_result[i]
                          << ", GPU value = " << gpu_result[i] << ", Absolute Difference = " << abs_diff << std::endl;
            }
            mismatch_count++;
        }
    }

    if (mismatch_count == 0) {
        std::cout << "All results match within the relative tolerance of " << relative_tolerance 
                  << " and absolute tolerance of " << absolute_tolerance << "." << std::endl;
    } else {
        std::cout << "Total mismatches: " << mismatch_count << " out of " << cpu_result.size() << " elements." << std::endl;
        std::cout << "Some results do not match within the specified tolerances." << std::endl;
    }
}

// void HeatEquationSolverNoStreams::solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) {
// #ifdef GGML_USE_CUDA
//     nvtxRangePush("GPU Solve");

//     dim3 grid_size((nx_ + block_size.x - 1) / block_size.x, (ny_ + block_size.y - 1) / block_size.y);

//     std::cout << "Using GPU without streams for computation." << std::endl;

//     for (int step = 0; step < num_steps; ++step) {
//         nvtxRangePush("GPU Step");

//         heat_equation_step_gpu(d_temp_, d_temp_next_, nx_, ny_, alpha_, dx_, dy_, dt_, kernelType, block_size, 0);

//         CUDA_CHECK(cudaDeviceSynchronize());
//         nvtxRangePop(); // End of GPU Step marker

//         std::swap(d_temp_, d_temp_next_);

//         if (step % 1000 == 0) {
//             nvtxRangePush("GPU Memory Copy to Host");
//             float center_temp;
//             CUDA_CHECK(cudaMemcpy(&center_temp, d_temp_ + nx_/2 * ny_ + ny_/2, sizeof(float), cudaMemcpyDeviceToHost));
//             std::cout << "GPU Step==nostream " << step << ", Center temp: " << center_temp << std::endl;
//             nvtxRangePop();
//         }
//     }

//     CUDA_CHECK(cudaMemcpy(temp_, d_temp_, nx_ * ny_ * sizeof(float), cudaMemcpyDeviceToHost));
//     nvtxRangePop(); // End of GPU Solve marker
// #else
//     std::cerr << "CUDA not available. Using CPU solver instead." << std::endl;
//     solve_cpu(num_steps);
// #endif
// }

std::vector<float> HeatEquationSolverNoStreams::get_temperature_field() const {
    std::vector<float> result(nx_ * ny_);
    memcpy(result.data(), temp_, nx_ * ny_ * sizeof(float));
    return result;
}

void HeatEquationSolverNoStreams::solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) {
#ifdef GGML_USE_CUDA
    nvtxRangePush("GPU Solve");

    dim3 grid_size((nx_ + block_size.x - 1) / block_size.x, (ny_ + block_size.y - 1) / block_size.y);

    std::cout << "Using GPU without streams for computation." << std::endl;
    std::cout << "Kernel Type: " << static_cast<int>(kernelType) << std::endl;
    std::cout << "Block Size: " << block_size.x << "x" << block_size.y << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        nvtxRangePush("GPU Step");

        // Call the heat_equation_step_gpu function from the CUDA file
        heat_equation_step_gpu(d_temp_, d_temp_next_, nx_, ny_, alpha_, dx_, dy_, dt_, kernelType, block_size, 0, 0);

        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop(); // End of GPU Step marker

        std::swap(d_temp_, d_temp_next_);

        if (step % 1000 == 0) {
            nvtxRangePush("GPU Memory Copy to Host");
            float center_temp;
            CUDA_CHECK(cudaMemcpy(&center_temp, d_temp_ + nx_/2 * ny_ + ny_/2, sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "GPU Step==nostream " << step << ", Center temp: " << center_temp << std::endl;
            nvtxRangePop();
        }
    }

    CUDA_CHECK(cudaMemcpy(temp_, d_temp_, nx_ * ny_ * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop(); // End of GPU Solve marker
#else
    std::cerr << "CUDA not available. Using CPU solver instead." << std::endl;
    solve_cpu(num_steps);
#endif
}