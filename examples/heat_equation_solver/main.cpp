#include "HeatEquationSolver.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

#ifdef GGML_USE_CUDA
#include <nvToolsExt.h>
#include <cuda_runtime.h>
#endif

int main(int argc, char** argv) {
    int nx = 200, ny = 200;
    float dx = 0.1f, dy = 0.1f, dt = 0.0001f, alpha = 0.1f;
    int num_steps = 100000;
    int num_threads = 1;  // Default to 1 thread

    // Default block size
    int block_dim_x = 16;
    int block_dim_y = 16;

    // Determine mode: CPU or GPU
    std::string mode = "cpu";  // Default to CPU
    if (argc > 1) {
        mode = argv[1];
    }

    std::string kernelTypeStr = "basic";
    if (argc > 2) {
        kernelTypeStr = argv[2];
    }

    if (mode == "cpu") {
        // For CPU mode, expect a third argument for the number of threads
        if (argc > 2) {
            try {
                num_threads = std::stoi(argv[2]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number of threads argument. Must be an integer." << std::endl;
                return -1;
            }
        }
    } else if (mode == "gpu") {
        // For GPU mode, expect third and fourth arguments for block dimensions
        if (argc > 3) {
            try {
                block_dim_x = std::stoi(argv[3]);
                block_dim_y = std::stoi(argv[4]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid block size arguments. Must be integers." << std::endl;
                return -1;
            }
        }

        // Check for maximum threads per block if in GPU mode
        int max_threads_per_block;
        cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
        if (block_dim_x * block_dim_y > max_threads_per_block) {
            std::cerr << "Error: Block size (" << block_dim_x << "x" << block_dim_y 
                      << ") exceeds the maximum threads per block (" 
                      << max_threads_per_block << ")." << std::endl;
            return -1;
        }
    }

    dim3 block_size(block_dim_x, block_dim_y);
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);

    printf("Block size: %d x %d\n", block_dim_x, block_dim_y);

#ifdef GGML_USE_CUDA
    nvtxMarkA("Program Start");
#endif

    HeatEquationSolver solver(nx, ny, dx, dy, dt, alpha, num_threads);

    // Set initial condition (e.g., a hot spot in the center)
    std::vector<float> initial_temp(nx * ny, 0.0f);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (std::abs(i - nx / 2) < nx / 10 && std::abs(j - ny / 2) < ny / 10) {
                initial_temp[i * ny + j] = 100.0f;
            }
        }
    }
    solver.set_initial_condition(initial_temp);

    std::vector<float> cpu_result;  // Declare cpu_result here

    if (mode == "cpu") {
        // Solve using CPU
#ifdef GGML_USE_CUDA
        nvtxRangePush("CPU Solve");
#endif
        auto start = std::chrono::high_resolution_clock::now();
        solver.solve_cpu(num_steps);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "CPU Solve Time: " << diff.count() << " s" << std::endl;
#ifdef GGML_USE_CUDA
        nvtxRangePop();
#endif

        // Get and print CPU results
        cpu_result = solver.get_temperature_field();
        std::cout << "CPU: Temperature at center: " << cpu_result[nx / 2 * ny + ny / 2] << std::endl;
    }

#ifdef GGML_USE_CUDA
    else if (mode == "gpu") {
        // Solve using GPU
        KernelType kernelType = KernelType::BASIC;
        if (kernelTypeStr == "shared") {
            kernelType = KernelType::SHARED_MEMORY;
        } else if (kernelTypeStr == "loop_unroll") {
            kernelType = KernelType::LOOP_UNROLL;
        }

        nvtxRangePush("GPU Solve");
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        solver.set_initial_condition(initial_temp);  // Reset initial condition
        solver.solve_gpu(num_steps, kernelType, block_size);  // Pass block size
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "GPU Solve Time: " << milliseconds << " ms" << std::endl;
        nvtxRangePop();

        // Get and print GPU results
        std::vector<float> gpu_result = solver.get_temperature_field();
        std::cout << "GPU: Temperature at center: " << gpu_result[nx / 2 * ny + ny / 2] << std::endl;
    }
#endif

#ifdef GGML_USE_CUDA
    nvtxMarkA("Program End");
#endif

    return 0;
}
