#ifndef HEAT_EQUATION_SOLVER_NO_STREAMS_H
#define HEAT_EQUATION_SOLVER_NO_STREAMS_H

#include "HeatEquationSolverBase.h"

class HeatEquationSolverNoStreams : public HeatEquationSolverBase {
public:
    HeatEquationSolverNoStreams(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads);
    
    // Explicitly declare the destructor
    ~HeatEquationSolverNoStreams() override;

    void set_initial_condition(const std::vector<float>& initial_temp) override;
    void solve_cpu(int num_steps) override;
    void solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) override;
    std::vector<float> get_temperature_field() const override;
    void verify_results(int num_steps, KernelType kernelType, dim3 block_size) override;

private:
#ifdef GGML_USE_CUDA
    float* d_temp_;
    float* d_temp_next_;
    int test_flag_;
#endif
};

#endif // HEAT_EQUATION_SOLVER_NO_STREAMS_H
