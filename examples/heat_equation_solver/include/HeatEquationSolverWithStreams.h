#ifndef HEAT_EQUATION_SOLVER_WITH_STREAMS_H
#define HEAT_EQUATION_SOLVER_WITH_STREAMS_H

#include "HeatEquationSolverBase.h"
#include <cuda_runtime.h>
#include <array>

class HeatEquationSolverWithStreams : public HeatEquationSolverBase {
public:
    HeatEquationSolverWithStreams(int nx, int ny, float dx, float dy, float dt, float alpha, int num_threads);
    ~HeatEquationSolverWithStreams();

    void set_initial_condition(const std::vector<float>& initial_temp) override;
    void solve_cpu(int num_steps) override;
    void solve_gpu(int num_steps, KernelType kernelType, dim3 block_size) override;
    std::vector<float> get_temperature_field() const override;
    void verify_results(int num_steps, KernelType kernelType, dim3 block_size) override;

private:
    static const int NUM_STREAMS = 2;
    std::array<cudaStream_t, NUM_STREAMS> streams_;
    float *d_temp_, *d_temp_next_;
    std::array<float*, NUM_STREAMS> d_temp_parts_, d_temp_next_parts_;
    int part_nx_, part_ny_;

    float* temp_cpu_ = nullptr;
    float* temp_next_cpu_ = nullptr;

    void update_overlapping_regions();
    void combine_partitions();
    void print_debug_info(int step,bool is_initial);
};

#endif // HEAT_EQUATION_SOLVER_WITH_STREAMS_H