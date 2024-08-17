#include "HeatEquationSolverFactory.h"
#include "HeatEquationSolverBase.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>

#ifdef GGML_USE_CUDA
#include <nvToolsExt.h>
#include <cuda_runtime.h>
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

int main(int argc, char** argv) {
    // Default parameters
    int nx = 128, ny = 128;
    float dx = 0.1f, dy = 0.1f, dt = 0.0001f, alpha = 0.1f;
    int num_steps = 10000;
    int num_threads = 12;  // Default number of threads
    int block_dim_x = 16, block_dim_y = 16;
    std::string mode = "cpu";
    std::string kernelTypeStr = "basic";

    KernelType kernelType = KernelType::BASIC;
    SolverType solverType = NO_STREAMS;

    // Parse command-line arguments
    if (argc > 1) {
        mode = argv[1];
    }

    if (mode == "gpu" && argc > 2) {
        kernelTypeStr = argv[2];
        std::transform(kernelTypeStr.begin(), kernelTypeStr.end(), kernelTypeStr.begin(), ::tolower);

        // Determine the kernel type and solver type based on the input
        if (kernelTypeStr == "basic") {
            kernelType = KernelType::BASIC;
        } else if (kernelTypeStr == "shared" || kernelTypeStr == "shared_memory") {
            kernelType = KernelType::SHARED_MEMORY;
        } else if (kernelTypeStr == "loop_unroll") {
            kernelType = KernelType::LOOP_UNROLL;
        } else if (kernelTypeStr == "fma") {
            kernelType = KernelType::FMA;
        } else if (kernelTypeStr == "shared_fma" || kernelTypeStr == "shared_memory_fma") {
            kernelType = KernelType::SHARED_MEMORY_FMA;
        } else if (kernelTypeStr == "loop_unroll_fma") {
            kernelType = KernelType::LOOP_UNROLL_FMA;
        } else if (kernelTypeStr == "two_stream" || kernelTypeStr == "twostreams" || kernelTypeStr == "two_streams") { 
            kernelType = KernelType::TWO_STREAM;
            solverType = WITH_STREAMS;

            // Ensure a default block size for two_streams if not provided
            if (argc <= 4) {
                block_dim_x = 8;
                block_dim_y = 8;
            }
        } else {
            std::cerr << "Invalid kernel type: " << kernelTypeStr << ". Using 'basic' as default." << std::endl;
            kernelType = KernelType::BASIC;
        }

        if (argc > 4) {
            block_dim_x = std::stoi(argv[3]);
            block_dim_y = std::stoi(argv[4]);
        }
    } else if (mode == "cpu") {
        if (argc > 2) {
            num_threads = std::stoi(argv[2]);
        }
    } else {
        std::cerr << "Invalid mode: " << mode << ". Must be 'cpu' or 'gpu'." << std::endl;
        return -1;
    }

    dim3 block_size(block_dim_x, block_dim_y);
    std::cout << "Block size: " << block_dim_x << " x " << block_dim_y << std::endl;

    // Create solver using factory method
    HeatEquationSolverBase* solver = create_solver(solverType, nx, ny, dx, dy, dt, alpha, num_threads);

    // Set initial conditions for both stream and non-stream versions
    std::vector<float> initial_temp(nx * ny, 0.0f);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (i == 0) {
                initial_temp[i * ny + j] = 1000.0f; // Top edge at 1000 degrees
            }
        }
    }
    solver->set_initial_condition(initial_temp);

    // Solve
    if (mode == "cpu") {
        auto start = std::chrono::high_resolution_clock::now();
        solver->solve_cpu(num_steps);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "CPU Solve Time: " << diff.count() << " s" << std::endl;

        std::vector<float> cpu_result = solver->get_temperature_field();
        std::cout << "CPU: Temperature at center: " << cpu_result[nx / 2 * ny + ny / 2] << std::endl;
    } else if (mode == "gpu") {
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Start the GPU timer
        CUDA_CHECK(cudaEventRecord(start));

        // Solve on GPU and verify results
        solver->verify_results(num_steps, kernelType, block_size);

        // Stop the GPU timer
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Calculate and print the elapsed time
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "Kernel Type: " << kernelTypeStr << std::endl;
        std::cout << "Total GPU Execution Time (including verification): " << milliseconds << " ms" << std::endl;

        // Retrieve and print the GPU result
        std::vector<float> gpu_result = solver->get_temperature_field();
        std::cout << "GPU: Temperature at center: " << gpu_result[nx / 2 * ny + ny / 2] << std::endl;

        // Clean up CUDA events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    delete solver;
    return 0;
}