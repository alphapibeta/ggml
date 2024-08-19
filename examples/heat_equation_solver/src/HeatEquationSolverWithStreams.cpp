
#include "HeatEquationSolverWithStreams.h"
#include "HeatEquationSolverNoStreams.h"
#include "HeatEquationKernels.h"
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

HeatEquationSolverWithStreams::HeatEquationSolverWithStreams(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads)
    : HeatEquationSolverBase(nx, ny, dx, dy, dt, alpha, num_threads) {
#ifdef USE_NVTX
    nvtxMarkA("HeatEquationSolverWithStreams Constructor");
#endif

    part_nx_ = nx_ / NUM_STREAMS;
    part_ny_ = ny_;

    // Allocate memory for the entire grid
    CUDA_CHECK(cudaMalloc(&d_temp_, nx_ * ny_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_next_, nx_ * ny_ * sizeof(float)));

    // Create streams and allocate memory for partitions
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        
        // Allocate memory for each partition (including overlap)
        int part_size = (part_nx_ + 2) * part_ny_ * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_temp_parts_[i], part_size));
        CUDA_CHECK(cudaMalloc(&d_temp_next_parts_[i], part_size));

        // Initialize partition arrays to zero
        CUDA_CHECK(cudaMemsetAsync(d_temp_parts_[i], 0, part_size, streams_[i]));
        CUDA_CHECK(cudaMemsetAsync(d_temp_next_parts_[i], 0, part_size, streams_[i]));
    }

    // Synchronize all streams after initialization
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }

    std::cout << "HeatEquationSolverWithStreams initialized with:" << std::endl;
    std::cout << "  Grid size: " << nx_ << " x " << ny_ << std::endl;
    std::cout << "  Partition size: " << part_nx_ << " x " << part_ny_ << std::endl;
    std::cout << "  Number of streams: " << NUM_STREAMS << std::endl;
}

HeatEquationSolverWithStreams::~HeatEquationSolverWithStreams() {
    CUDA_CHECK(cudaFree(d_temp_));
    CUDA_CHECK(cudaFree(d_temp_next_));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(d_temp_parts_[i]));
        CUDA_CHECK(cudaFree(d_temp_next_parts_[i]));
        CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
    if (temp_cpu_) delete[] temp_cpu_;
    if (temp_next_cpu_) delete[] temp_next_cpu_;
}

// void HeatEquationSolverWithStreams::set_initial_condition(const std::vector<float>& initial_temp){
//     if (initial_temp.size() != static_cast<size_t>(nx_ * ny_)) {
//         throw std::invalid_argument("Initial temperature vector size does not match grid size");
//     }

//     // Initialize the top row of the full grid (y=0) in the main grid memory `d_temp_`
//     std::vector<float> modified_initial_temp(nx_ * ny_, 0.0f);
//     for (int x = 0; x < nx_; ++x) {
//         modified_initial_temp[x] = 1000.0f; // Set the top row to 1000 across all x
//     }

//     // Copy the modified initial temperature to the GPU memory for the entire grid
//     CUDA_CHECK(cudaMemcpy(d_temp_, modified_initial_temp.data(), nx_ * ny_ * sizeof(float), cudaMemcpyHostToDevice));

//     // Now initialize each partition's memory, including ghost cells
//     for (int i = 0; i < NUM_STREAMS; ++i) {
//         int offset = i * part_nx_; // Offset to the start of this partition in the x-dimension// Initialize the main data region for this partition (excluding ghost cells)
//         for (int x = 0; x < part_nx_; ++x) {
//             CUDA_CHECK(cudaMemcpyAsync(&d_temp_parts_[i][x + 1], &modified_initial_temp[offset + x], sizeof(float), 
//                                        cudaMemcpyHostToDevice, streams_[i]));
//         }

//         // Initialize the ghost cells
//         if (i > 0) {
//             // Set left ghost cell of current partition from the right edge of the previous partition
//             CUDA_CHECK(cudaMemcpyAsync(&d_temp_parts_[i][0], &modified_initial_temp[offset - 1], sizeof(float), 
//                                        cudaMemcpyHostToDevice, streams_[i]));
//         }

//         if (i < NUM_STREAMS - 1) {
//             // Set right ghost cell of current partition from the left edge of the next partition
//             CUDA_CHECK(cudaMemcpyAsync(&d_temp_parts_[i][part_nx_ + 1], &modified_initial_temp[offset + part_nx_], sizeof(float), 
//                                        cudaMemcpyHostToDevice, streams_[i]));
//         }
//     }

//     // Synchronize all streams to ensure initialization is complete
//     for (int i = 0; i < NUM_STREAMS; ++i) {
//         CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
//     }

//     // Combine the partitions back into the main grid to ensure consistency
//     combine_partitions();

//     // Debug information (optional)
//     print_debug_info(0, true);
// }



void HeatEquationSolverWithStreams::set_initial_condition(const std::vector<float>& initial_temp){
    if (initial_temp.size() != static_cast<size_t>(nx_ * ny_)) {
        throw std::invalid_argument("Initial temperature vector size does not match grid size");
    }


    std::memcpy(temp_cpu_, initial_temp.data(), nx_ * ny_ * sizeof(float));

    for (int i = 0; i < nx_; ++i) {
        temp_cpu_[i * ny_] = 1000.0f;
    }


    CUDA_CHECK(cudaMemcpy(d_temp_, temp_cpu_, nx_ * ny_ * sizeof(float), cudaMemcpyHostToDevice));


    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start_x = i * part_nx_;
        int end_x = (i == NUM_STREAMS - 1) ? nx_ : (i + 1) * part_nx_;
        int width = end_x - start_x;


        CUDA_CHECK(cudaMemcpy2DAsync(d_temp_parts_[i] + 1, (part_nx_ + 2) * sizeof(float),
                                     d_temp_ + start_x, nx_ * sizeof(float),
                                     width * sizeof(float), ny_,
                                     cudaMemcpyDeviceToDevice, streams_[i]));


        if (i > 0) {
            partition_data[0] = top_row[offset - 1]; // Left ghost cell from previous partition
        }
        if (i < NUM_STREAMS - 1) {
            partition_data[part_nx_ + 1] = top_row[offset + part_nx_]; // Right ghost cell from next partition
        }

        // Copy the partition data to the GPU memory for this partition
        CUDA_CHECK(cudaMemcpyAsync(d_temp_parts_[i], partition_data.data(), (part_nx_ + 2) * sizeof(float), cudaMemcpyHostToDevice, streams_[i]));
    }

    // Synchronize all streams to ensure initialization is complete
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }

    // Combine the partitions back into the main grid to ensure consistency
    combine_partitions();
    print_debug_info(0, true, true);
}



void HeatEquationSolverWithStreams::set_initial_condition_cpu(const std::vector<float>& initial_temp) {
    std::memcpy(temp_cpu_, initial_temp.data(), nx_ * ny_ * sizeof(float));

    for (int i = 0; i < ny_; ++i) {
        temp_cpu_[i * nx_] = 1000.0f;
    }
}






void HeatEquationSolverWithStreams::set_initial_condition(const std::vector<float>& initial_temp) {
    set_initial_condition_gpu(initial_temp);
    set_initial_condition_cpu(initial_temp);
}





void HeatEquationSolverWithStreams::solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) {
#ifdef USE_NVTX
    nvtxRangePush("GPU Solve With Streams");
#endif

    dim3 grid_size((part_nx_ + 2 + block_size.x - 1) / block_size.x, (ny_ + block_size.y - 1) / block_size.y);

    std::cout << "Using GPU with streams for computation." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
#ifdef USE_NVTX
        nvtxRangePush("GPU Step");
#endif

        for (int i = 0; i < NUM_STREAMS; ++i) {
            void* args[] = {&d_temp_parts_[i], &d_temp_next_parts_[i], &nx_, &ny_, &alpha_, &dx_, &dy_, &dt_, &i};
            CUDA_CHECK(cudaLaunchKernel((void*)heat_equation_kernel_two_stream, grid_size, block_size, args, 0, streams_[i]));
        }

        // Synchronize all streams
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
        }

        // Swap pointers for each partition
        for (int i = 0; i < NUM_STREAMS; ++i) {
            std::swap(d_temp_parts_[i], d_temp_next_parts_[i]);
        }

        // Update overlapping regions
        update_overlapping_regions();

        if (step % 1000 == 0 ) {
            // Combine partitions into the main grid
            combine_partitions();

            // Print center temperature
            float center_temp;
            CUDA_CHECK(cudaMemcpy(&center_temp, d_temp_ + (nx_ / 2) * ny_ + (ny_ / 2), sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "GPU Step==streams " << step << ", Center temp: " << center_temp << std::endl;
        }

#ifdef USE_NVTX
        nvtxRangePop(); // End of GPU Step marker
#endif

    for (int step = 0; step < num_steps; ++step) {
#ifdef _OPENMP
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


#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < nx_; ++i) {
            u_next[i * ny_] = 1000.0f; 
            u_next[i * ny_ + ny_ - 1] = u[i * ny_ + ny_ - 1]; 
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < ny_; ++j) {
            u_next[j] = u[j]; 
            u_next[(nx_ - 1) * ny_ + j] = u[(nx_ - 1) * ny_ + j]; 
        }

        std::swap(u, u_next);
        std::swap(temp_cpu_, temp_next_cpu_);

        if (step % 1000 == 0) {
            std::cout << "CPU Step " << step << ", Center temp: " << u[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
        }
    }

    std::memcpy(temp_, temp_cpu_, nx_ * ny_ * sizeof(float));
}


void HeatEquationSolverWithStreams::update_overlapping_regions() {
    for (int i = 0; i < NUM_STREAMS - 1; ++i) {

        CUDA_CHECK(cudaMemcpy2DAsync(d_temp_parts_[i + 1], (part_nx_ + 2) * sizeof(float),
                                     d_temp_parts_[i] + part_nx_, (part_nx_ + 2) * sizeof(float),
                                     sizeof(float), ny_,
                                     cudaMemcpyDeviceToDevice, streams_[i + 1]));


        CUDA_CHECK(cudaMemcpy2DAsync(d_temp_parts_[i] + part_nx_ + 1, (part_nx_ + 2) * sizeof(float),
                                     d_temp_parts_[i + 1] + 1, (part_nx_ + 2) * sizeof(float),
                                     sizeof(float), ny_,
                                     cudaMemcpyDeviceToDevice, streams_[i]));
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }
}

void HeatEquationSolverWithStreams::combine_partitions() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * part_nx_ * ny_;
        CUDA_CHECK(cudaMemcpyAsync(d_temp_ + offset, d_temp_parts_[i] + ny_, 
                                   part_nx_ * ny_ * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, streams_[i]));
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }
}



void HeatEquationSolverWithStreams::reinforce_boundary_conditions() {
    float boundary_value = 1000.0f;


    for (int i = 0; i < NUM_STREAMS; ++i) {

        CUDA_CHECK(cudaMemcpyAsync(d_temp_parts_[i], &boundary_value, part_nx_ * sizeof(float), cudaMemcpyHostToDevice, streams_[i]));
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }


    for (int i = 0; i < NUM_STREAMS - 1; ++i) {
        CUDA_CHECK(cudaMemcpy2DAsync(d_temp_parts_[i + 1], (part_nx_ + 2) * sizeof(float),
                                     d_temp_parts_[i] + part_nx_, (part_nx_ + 2) * sizeof(float),
                                     sizeof(float), ny_,
                                     cudaMemcpyDeviceToDevice, streams_[i + 1]));

        CUDA_CHECK(cudaMemcpy2DAsync(d_temp_parts_[i] + part_nx_ + 1, (part_nx_ + 2) * sizeof(float),
                                     d_temp_parts_[i + 1] + 1, (part_nx_ + 2) * sizeof(float),
                                     sizeof(float), ny_,
                                     cudaMemcpyDeviceToDevice, streams_[i]));
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }
}



void HeatEquationSolverWithStreams::print_debug_info(int step, bool is_initial, bool use_gpu = true) {
    std::cout << (is_initial ? "Initial Condition" : "Temperature field at step " + std::to_string(step)) << ":" << std::endl;

    if (use_gpu) {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            std::vector<float> part_temp((part_nx_ + 2) * part_ny_);
            CUDA_CHECK(cudaMemcpy(part_temp.data(), d_temp_parts_[i], (part_nx_ + 2) * part_ny_ * sizeof(float), cudaMemcpyDeviceToHost));

            std::cout << "Partition " << i << " (GPU):" << std::endl;
            for (int y = 0; y < std::min(10, part_ny_); ++y) {
                for (int x = 0; x < part_nx_ + 2; ++x) {
                    int global_x = (i * part_nx_) + (x - 1);
                    if (global_x >= 0 && global_x < nx_) {
                        std::cout << std::setw(10) << std::fixed << std::setprecision(4) << part_temp[y * (part_nx_ + 2) + x] << " ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    } else {

        for (int i = 0; i < NUM_STREAMS; ++i) {
            std::cout << "Partition " << i << " (CPU):" << std::endl;
            for (int y = 0; y < std::min(10, part_ny_); ++y) {
                for (int x = 0; x < part_nx_ + 2; ++x) {
                    int global_x = (i * part_nx_) + (x - 1);
                    if (global_x >= 0 && global_x < nx_) {
                        int idx = y * nx_ + global_x;
                        std::cout << std::setw(10) << std::fixed << std::setprecision(4) << temp_cpu_[idx] << " ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}


std::vector<float> HeatEquationSolverWithStreams::get_temperature_field_gpu() const {
    std::vector<float> result(nx_ * ny_);
    CUDA_CHECK(cudaMemcpy(result.data(), d_temp_, nx_ * ny_ * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

void HeatEquationSolverWithStreams::verify_results(int num_steps, KernelType kernelType, dim3 block_size) {
    std::cout << "Verifying results between CPU and GPU with streams..." << std::endl;

    // Generate a proper initial condition
    std::vector<float> initial_condition(nx_ * ny_, 0.0f);
    for (int i = 0; i < nx_; ++i) {
        initial_condition[i * ny_] = 1000.0f;  
    }


    set_initial_condition_gpu(initial_condition);
    set_initial_condition_cpu(initial_condition);


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

    // ... (rest of the function remains the same)
}


void HeatEquationSolverWithStreams::solve_cpu(int num_steps) {
    // Allocate memory for CPU computation if not already done
    if (!temp_cpu_ || !temp_next_cpu_) {
        temp_cpu_ = new float[nx_ * ny_];
        temp_next_cpu_ = new float[nx_ * ny_];
        
        // Initialize temp_cpu_ with the initial condition
        std::vector<float> initial_condition = get_temperature_field();
        std::memcpy(temp_cpu_, initial_condition.data(), nx_ * ny_ * sizeof(float));
    }

    float* u = temp_cpu_;
    float* u_next = temp_next_cpu_;

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
        std::swap(temp_cpu_, temp_next_cpu_);

        if (step % 1000 == 0) {
            std::cout << "CPU Step " << step << ", Center temp: " << u[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
        }
#ifdef USE_NVTX
        nvtxRangePop();
#endif
    }

    // Copy the final result to the main temperature field
    std::memcpy(temp_, temp_cpu_, nx_ * ny_ * sizeof(float));
}