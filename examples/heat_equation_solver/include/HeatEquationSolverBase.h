#ifndef HEAT_EQUATION_SOLVER_BASE_H
#define HEAT_EQUATION_SOLVER_BASE_H

#include "ggml.h"
#include <vector>
#include <string>
#include <cmath>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

enum class KernelType {
    BASIC,
    SHARED_MEMORY,
    LOOP_UNROLL,
    FMA,
    SHARED_MEMORY_FMA,
    LOOP_UNROLL_FMA,
    TWO_STREAM
};

class HeatEquationSolverBase {
public:
    HeatEquationSolverBase(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads)
        : nx_(nx), ny_(ny), dx_(dx), dy_(dy), dt_(dt), alpha_(alpha), num_threads_(num_threads) {
        temp_ = new float[nx_ * ny_];
        temp_next_ = new float[nx_ * ny_];
    }

    virtual ~HeatEquationSolverBase() {
        delete[] temp_;
        delete[] temp_next_;
    }

    virtual void set_initial_condition(const std::vector<float>& initial_temp) = 0;
    virtual void solve_cpu(int num_steps) = 0;
    virtual void solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) = 0;
    virtual std::vector<float> get_temperature_field() const = 0;
    virtual void verify_results(int num_steps, KernelType kernelType, dim3 block_size) = 0;

protected:
    int nx_, ny_;
    float dx_, dy_, dt_, alpha_;
    int num_threads_;
    float* temp_;
    float* temp_next_;
};

#endif // HEAT_EQUATION_SOLVER_BASE_H