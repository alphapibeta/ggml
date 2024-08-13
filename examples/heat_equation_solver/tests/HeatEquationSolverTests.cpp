#include "HeatEquationSolver.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

void test_cpu_solver(int num_threads) {
    std::cout << "Running CPU solver test with " << num_threads << " threads...\n";

    int nx = 200, ny = 200;
    float dx = 0.1f, dy = 0.1f, dt = 0.0001f, alpha = 0.1f;
    int num_steps = 1000;

    HeatEquationSolver solver(nx, ny, dx, dy, dt, alpha, num_threads);

    // Set initial condition
    std::vector<float> initial_temp(nx * ny, 0.0f);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (std::abs(i - nx / 2) < nx / 10 && std::abs(j - ny / 2) < ny / 10) {
                initial_temp[i * ny + j] = 100.0f;
            }
        }
    }
    solver.set_initial_condition(initial_temp);

    // Run solver
    solver.solve_cpu(num_steps);

    // Get results
    std::vector<float> result = solver.get_temperature_field();
    float center_temp = result[nx / 2 * ny + ny / 2];
    std::cout << "CPU solver center temperature: " << center_temp << "\n";

    // Verify result (just a simple range check for this example)
    assert(center_temp > 0 && center_temp < 100);
}

void test_basic_kernel_gpu() {
    std::cout << "Running basic GPU kernel test...\n";

    int nx = 200, ny = 200;
    float dx = 0.1f, dy = 0.1f, dt = 0.0001f, alpha = 0.1f;
    int num_steps = 1000;

    // Block size
    int block_dim_x = 16;
    int block_dim_y = 16;
    dim3 block_size(block_dim_x, block_dim_y);

    HeatEquationSolver solver(nx, ny, dx, dy, dt, alpha);

    // Set initial condition
    std::vector<float> initial_temp(nx * ny, 0.0f);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (std::abs(i - nx / 2) < nx / 10 && std::abs(j - ny / 2) < ny / 10) {
                initial_temp[i * ny + j] = 100.0f;
            }
        }
    }
    solver.set_initial_condition(initial_temp);

    // Run solver
    solver.solve_gpu(num_steps, KernelType::BASIC, block_size);

    // Get results
    std::vector<float> result = solver.get_temperature_field();
    float center_temp = result[nx / 2 * ny + ny / 2];
    std::cout << "Basic GPU kernel center temperature: " << center_temp << "\n";

    // Verify result (just a simple range check for this example)
    assert(center_temp > 0 && center_temp < 100);
}

void test_shared_memory_kernel_gpu() {
    std::cout << "Running shared memory GPU kernel test...\n";

    int nx = 200, ny = 200;
    float dx = 0.1f, dy = 0.1f, dt = 0.0001f, alpha = 0.1f;
    int num_steps = 1000;

    // Block size
    int block_dim_x = 16;
    int block_dim_y = 16;
    dim3 block_size(block_dim_x, block_dim_y);

    HeatEquationSolver solver(nx, ny, dx, dy, dt, alpha);

    // Set initial condition
    std::vector<float> initial_temp(nx * ny, 0.0f);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (std::abs(i - nx / 2) < nx / 10 && std::abs(j - ny / 2) < ny / 10) {
                initial_temp[i * ny + j] = 100.0f;
            }
        }
    }
    solver.set_initial_condition(initial_temp);

    // Run solver
    solver.solve_gpu(num_steps, KernelType::SHARED_MEMORY, block_size);

    // Get results
    std::vector<float> result = solver.get_temperature_field();
    float center_temp = result[nx / 2 * ny + ny / 2];
    std::cout << "Shared memory GPU kernel center temperature: " << center_temp << "\n";

    // Verify result
    assert(center_temp > 0 && center_temp < 100);
}

void test_loop_unroll_kernel_gpu() {
    std::cout << "Running loop unroll GPU kernel test...\n";

    int nx = 200, ny = 200;
    float dx = 0.1f, dy = 0.1f, dt = 0.0001f, alpha = 0.1f;
    int num_steps = 1000;

    // Block size
    int block_dim_x = 16;
    int block_dim_y = 16;
    dim3 block_size(block_dim_x, block_dim_y);

    HeatEquationSolver solver(nx, ny, dx, dy, dt, alpha);

    // Set initial condition
    std::vector<float> initial_temp(nx * ny, 0.0f);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (std::abs(i - nx / 2) < nx / 10 && std::abs(j - ny / 2) < ny / 10) {
                initial_temp[i * ny + j] = 100.0f;
            }
        }
    }
    solver.set_initial_condition(initial_temp);

    // Run solver
    solver.solve_gpu(num_steps, KernelType::LOOP_UNROLL, block_size);

    // Get results
    std::vector<float> result = solver.get_temperature_field();
    float center_temp = result[nx / 2 * ny + ny / 2];
    std::cout << "Loop unroll GPU kernel center temperature: " << center_temp << "\n";

    // Verify result
    assert(center_temp > 0 && center_temp < 100);
}

void run_tests() {
    // Test CPU solver with different thread counts
    test_cpu_solver(1);
    test_cpu_solver(4);
    test_cpu_solver(8);
    test_cpu_solver(12);

    // Test different GPU kernels
    test_basic_kernel_gpu();
    test_shared_memory_kernel_gpu();
    test_loop_unroll_kernel_gpu();

    std::cout << "All tests passed.\n";
}

int main() {
    run_tests();
    return 0;
}
