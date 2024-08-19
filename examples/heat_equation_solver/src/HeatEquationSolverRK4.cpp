#include "HeatEquationSolverRK4.h"
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

HeatEquationSolverRK4::HeatEquationSolverRK4(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads)
    : HeatEquationSolverBase(nx, ny, dx, dy, dt, alpha, num_threads) {
#ifdef USE_NVTX
    nvtxMarkA("HeatEquationSolverRK4 Constructor");
#endif

    part_nx_ = nx_ / NUM_STREAMS;
    part_ny_ = ny_;

#ifdef USE_NVTX
    nvtxRangePushA("Allocate memory for the entire grid");
#endif

    CUDA_CHECK(cudaMalloc(&d_temp_, nx_ * ny_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_next_, nx_ * ny_ * sizeof(float)));
#ifdef USE_NVTX
    nvtxRangePop();
#endif

#ifdef USE_NVTX
    nvtxRangePushA("Allocate memory for RK4 stages");
#endif

    for (int i = 0; i < 4; ++i) {
        CUDA_CHECK(cudaMalloc(&d_k_[i], nx_ * ny_ * sizeof(float)));
    }
#ifdef USE_NVTX
    nvtxRangePop();
#endif

#ifdef USE_NVTX
    nvtxRangePushA("Create streams and allocate memory for partitions");
#endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        

        int part_size = (part_nx_ + 2) * part_ny_ * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_temp_parts_[i], part_size));
        CUDA_CHECK(cudaMalloc(&d_temp_next_parts_[i], part_size));


        CUDA_CHECK(cudaMemsetAsync(d_temp_parts_[i], 0, part_size, streams_[i]));
        CUDA_CHECK(cudaMemsetAsync(d_temp_next_parts_[i], 0, part_size, streams_[i]));
    }
#ifdef USE_NVTX
    nvtxRangePop();
#endif


    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }

    std::cout << "HeatEquationSolverRK4 initialized with:" << std::endl;
    std::cout << "  Grid size: " << nx_ << " x " << ny_ << std::endl;
    std::cout << "  Partition size: " << part_nx_ << " x " << part_ny_ << std::endl;
    std::cout << "  Number of streams: " << NUM_STREAMS << std::endl;
}


HeatEquationSolverRK4::~HeatEquationSolverRK4() {
    CUDA_CHECK(cudaFree(d_temp_));
    CUDA_CHECK(cudaFree(d_temp_next_));

    for (int i = 0; i < 4; ++i) {
        CUDA_CHECK(cudaFree(d_k_[i]));
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(d_temp_parts_[i]));
        CUDA_CHECK(cudaFree(d_temp_next_parts_[i]));
        CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
    if (temp_cpu_) delete[] temp_cpu_;
    if (temp_next_cpu_) delete[] temp_next_cpu_;
}

void HeatEquationSolverRK4::set_initial_condition(const std::vector<float>& initial_temp) {
    if (initial_temp.size() != static_cast<size_t>(nx_ * ny_)) {
        throw std::invalid_argument("Initial temperature vector size does not match grid size");
    }


    std::vector<float> modified_initial_temp(nx_ * ny_, 0.0f);
    for (int x = 0; x < nx_; ++x) {
        modified_initial_temp[x] = 1000.0f; 
    }


    CUDA_CHECK(cudaMemcpy(d_temp_, modified_initial_temp.data(), nx_ * ny_ * sizeof(float), cudaMemcpyHostToDevice));


    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * part_nx_; 
        
        for (int x = 0; x < part_nx_; ++x) {
            CUDA_CHECK(cudaMemcpyAsync(&d_temp_parts_[i][x + 1], &modified_initial_temp[offset + x], sizeof(float), 
                                       cudaMemcpyHostToDevice, streams_[i]));
        }


        if (i > 0) {

            CUDA_CHECK(cudaMemcpyAsync(&d_temp_parts_[i][0], &modified_initial_temp[offset - 1], sizeof(float), 
                                       cudaMemcpyHostToDevice, streams_[i]));
        }

        if (i < NUM_STREAMS - 1) {

            CUDA_CHECK(cudaMemcpyAsync(&d_temp_parts_[i][part_nx_ + 1], &modified_initial_temp[offset + part_nx_], sizeof(float), 
                                       cudaMemcpyHostToDevice, streams_[i]));
        }
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }


    combine_partitions();
}

void HeatEquationSolverRK4::solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) {
#ifdef USE_NVTX
    nvtxRangePush("GPU Solve RK4 With Streams");
#endif

    std::cout << "Using GPU with RK4 and streams for computation." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
#ifdef USE_NVTX
        nvtxRangePush("GPU RK4 Step");
#endif

        for (int i = 0; i < NUM_STREAMS; ++i) {
            heat_equation_step_rk4_gpu(d_temp_parts_[i], d_temp_next_parts_[i], 
                                       d_k_[0], d_k_[1], d_k_[2], d_k_[3],
                                       part_nx_, part_ny_, alpha_, dx_, dy_, dt_,
                                       block_size, streams_[i], i);
        }


        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
        }


        for (int i = 0; i < NUM_STREAMS; ++i) {
            std::swap(d_temp_parts_[i], d_temp_next_parts_[i]);
        }


        update_overlapping_regions();

        if (step % 1000 == 0) {

            combine_partitions();


            float center_temp;
            CUDA_CHECK(cudaMemcpy(&center_temp, d_temp_ + (nx_ / 2) * ny_ + (ny_ / 2), sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "GPU Step RK4 " << step << ", Center temp: " << center_temp << std::endl;
        }

#ifdef USE_NVTX
        nvtxRangePop(); 
#endif
    }


    combine_partitions();

#ifdef USE_NVTX
    nvtxRangePop(); // End of GPU Solve RK4 With Streams marker
#endif
}



void HeatEquationSolverRK4::rk4_step_gpu(dim3 grid_size, dim3 block_size) {
    float dt_half = dt_ / 2.0f;
    float dt_sixth = dt_ / 6.0f;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        heat_equation_step_rk4_gpu_wrapper(
            d_temp_parts_[i], d_temp_next_parts_[i],
            d_k_[0], d_k_[1], d_k_[2], d_k_[3],
            part_nx_, part_ny_, alpha_, dx_, dy_, dt_,
            grid_size, block_size, streams_[i], i
        );
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        std::swap(d_temp_parts_[i], d_temp_next_parts_[i]);
    }


    update_overlapping_regions();
}

void HeatEquationSolverRK4::update_overlapping_regions() {
    for (int i = 1; i < NUM_STREAMS; ++i) {

        CUDA_CHECK(cudaMemcpyAsync(d_temp_parts_[i], 
                                   d_temp_parts_[i-1] + (part_nx_) * ny_, 
                                   ny_ * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, 
                                   streams_[i]));


        CUDA_CHECK(cudaMemcpyAsync(d_temp_parts_[i-1] + (part_nx_ + 1) * ny_, 
                                   d_temp_parts_[i] + ny_, 
                                   ny_ * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, 
                                   streams_[i-1]));
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }
}

void HeatEquationSolverRK4::combine_partitions() {
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

std::vector<float> HeatEquationSolverRK4::get_temperature_field() const {
    std::vector<float> result(nx_ * ny_);
    CUDA_CHECK(cudaMemcpy(result.data(), d_temp_, nx_ * ny_ * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

void HeatEquationSolverRK4::solve_cpu(int num_steps) {
#ifdef USE_NVTX
    nvtxRangePush("CPU RK4 Solve");
#endif


    if (!temp_cpu_ || !temp_next_cpu_) {
        temp_cpu_ = new float[nx_ * ny_];
        temp_next_cpu_ = new float[nx_ * ny_];
        

        std::vector<float> initial_condition = get_temperature_field();
        std::memcpy(temp_cpu_, initial_condition.data(), nx_ * ny_ * sizeof(float));
    }


    float* k1 = new float[nx_ * ny_];
    float* k2 = new float[nx_ * ny_];
    float* k3 = new float[nx_ * ny_];
    float* k4 = new float[nx_ * ny_];

#ifdef USE_OPENMP
    omp_set_num_threads(num_threads_);
    std::cout << "Using OpenMP with " << num_threads_ << " threads for CPU computation." << std::endl;
#endif

    for (int step = 0; step < num_steps; ++step) {
#ifdef USE_NVTX
        nvtxRangePush("CPU RK4 Step");
#endif


        rk4_stage_cpu(temp_cpu_, k1, 0.0f);
        rk4_stage_cpu(temp_cpu_, k2, 0.5f * dt_, k1);
        rk4_stage_cpu(temp_cpu_, k3, 0.5f * dt_, k2);
        rk4_stage_cpu(temp_cpu_, k4, dt_, k3);


#ifdef USE_OPENMP
        #pragma omp parallel for collapse(2)
#endif
        for (int i = 1; i < nx_ - 1; ++i) {
            for (int j = 1; j < ny_ - 1; ++j) {
                int idx = i * ny_ + j;
                temp_next_cpu_[idx] = temp_cpu_[idx] + 
                    (dt_ / 6.0f) * (k1[idx] + 2.0f * k2[idx] + 2.0f * k3[idx] + k4[idx]);
            }
        }

        
#ifdef USE_OPENMP
        #pragma omp parallel for
#endif
        for (int i = 0; i < nx_; ++i) {
            temp_next_cpu_[i * ny_] = temp_cpu_[i * ny_];
            temp_next_cpu_[i * ny_ + ny_ - 1] = temp_cpu_[i * ny_ + ny_ - 1];
        }
#ifdef USE_OPENMP
        #pragma omp parallel for
#endif
        for (int j = 0; j < ny_; ++j) {
            temp_next_cpu_[j] = temp_cpu_[j];
            temp_next_cpu_[(nx_ - 1) * ny_ + j] = temp_cpu_[(nx_ - 1) * ny_ + j];
        }

        std::swap(temp_cpu_, temp_next_cpu_);

        if (step % 1000 == 0) {
            std::cout << "CPU RK4 Step " << step << ", Center temp: " << temp_cpu_[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
        }

#ifdef USE_NVTX
        nvtxRangePop(); 
#endif
    }


    std::memcpy(temp_, temp_cpu_, nx_ * ny_ * sizeof(float));

    // Clean up
    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] k4;

#ifdef USE_NVTX
    nvtxRangePop(); // End of CPU RK4 Solve
#endif
}

void HeatEquationSolverRK4::rk4_stage_cpu(const float* temp, float* k, float dt_factor, const float* prev_k) {
#ifdef USE_NVTX
    nvtxRangePush("CPU RK4 Stage");
#endif

#ifdef USE_OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 1; i < nx_ - 1; ++i) {
        for (int j = 1; j < ny_ - 1; ++j) {
            int idx = i * ny_ + j;
            float temp_val = temp[idx];
            if (prev_k) {
                temp_val += dt_factor * prev_k[idx];
            }

            float laplacian = 
                (temp[idx - ny_] + temp[idx + ny_] - 2 * temp_val) / (dx_ * dx_) +
                (temp[idx - 1] + temp[idx + 1] - 2 * temp_val) / (dy_ * dy_);

            k[idx] = alpha_ * laplacian;
        }
    }

#ifdef USE_NVTX
    nvtxRangePop(); // End of CPU RK4 Stage
#endif
}

void HeatEquationSolverRK4::verify_results(int num_steps, KernelType kernelType, dim3 block_size) {
    std::cout << "Verifying results between CPU and GPU with RK4..." << std::endl;

    // Store the initial condition
    std::vector<float> initial_condition = get_temperature_field();

    // Solve on GPU with RK4 (only once)
    auto gpu_start = std::chrono::high_resolution_clock::now();
    solve_gpu(num_steps, kernelType, block_size);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_diff = gpu_end - gpu_start;

    std::vector<float> gpu_result = get_temperature_field();

    // Print GPU center temperature and execution time
    std::cout << "GPU RK4: Temperature at center: " << gpu_result[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
    std::cout << "GPU RK4 Execution Time: " << gpu_diff.count() << " ms" << std::endl;

    // // Reset the initial condition for CPU
    // set_initial_condition(initial_condition);

    // // Solve on CPU (if implemented)
    // auto cpu_start = std::chrono::high_resolution_clock::now();
    // solve_cpu(num_steps);
    // auto cpu_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> cpu_diff = cpu_end - cpu_start;

    // std::vector<float> cpu_result = get_temperature_field();

    // // Print CPU center temperature and execution time
    // std::cout << "CPU RK4: Temperature at center: " << cpu_result[nx_ / 2 * ny_ + ny_ / 2] << std::endl;
    // std::cout << "CPU RK4 Execution Time: " << cpu_diff.count() << " ms" << std::endl;

    // // Compare results
    // float max_diff = 0.0f;
    // float avg_diff = 0.0f;
    // int num_large_diff = 0;
    // const float large_diff_threshold = 1e-5f;

    // for (int i = 0; i < nx_ * ny_; ++i) {
    //     float diff = std::abs(gpu_result[i] - cpu_result[i]);
    //     max_diff = std::max(max_diff, diff);
    //     avg_diff += diff;
    //     if (diff > large_diff_threshold) {
    //         ++num_large_diff;
    //     }
    // }

    // avg_diff /= (nx_ * ny_);

    // std::cout << "Comparison results:" << std::endl;
    // std::cout << "  Maximum difference: " << max_diff << std::endl;
    // std::cout << "  Average difference: " << avg_diff << std::endl;
    // std::cout << "  Number of large differences: " << num_large_diff << std::endl;

    // if (max_diff < large_diff_threshold) {
    //     std::cout << "Verification PASSED: GPU and CPU results match within tolerance." << std::endl;
    // } else {
    //     std::cout << "Verification FAILED: GPU and CPU results have significant differences." << std::endl;
    // }

    // // Performance comparison
    // double speedup = cpu_diff.count() / gpu_diff.count();
    // std::cout << "GPU speedup over CPU: " << speedup << "x" << std::endl;
}