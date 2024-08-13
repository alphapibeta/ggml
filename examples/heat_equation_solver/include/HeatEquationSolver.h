#ifndef HEAT_EQUATION_SOLVER_H
#define HEAT_EQUATION_SOLVER_H

#include "ggml.h"
#include <vector>
#include <string>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>  // Include CUDA runtime header
#include <nvToolsExt.h>
#endif

enum class KernelType {
    BASIC,
    SHARED_MEMORY,
    LOOP_UNROLL
};

#ifdef GGML_USE_CUDA
extern "C" void heat_equation_step_gpu(float* u, float* u_next, int nx, int ny, float alpha, float dx, float dy, float dt, KernelType kernelType, dim3 block_size);
#endif

class HeatEquationSolver {
public:
    HeatEquationSolver(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads = 1);
    ~HeatEquationSolver();

    void set_initial_condition(const std::vector<float>& initial_temp);
    void solve_cpu(int num_steps);
    
#ifdef GGML_USE_CUDA
    void solve_gpu(int num_steps, KernelType kernelType, dim3 block_size);
#endif
    
    std::vector<float> get_temperature_field() const;

private:
    int nx_, ny_;
    float dx_, dy_, dt_, alpha_;
    int num_threads_;  // Number of threads for OpenMP

    struct ggml_context* ctx_;
    struct ggml_tensor* temp_;
    struct ggml_tensor* temp_next_;

#ifdef GGML_USE_CUDA
    float* d_temp_;
    float* d_temp_next_;
#endif

    void swap_tensors();
};

#endif // HEAT_EQUATION_SOLVER_H
