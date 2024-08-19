#!/bin/bash

output_file="heat_equation_kernels.txt"
> "$output_file"

# List of kernel names and their corresponding GPU solver types
declare -A kernels
kernels["heat_equation_kernel_shared_memory_fma"]="shared_fma"
kernels["heat_equation_kernel_loop_unroll_fma"]="loop_unroll_fma"
kernels["heat_equation_kernel_shared_memory"]="shared"
kernels["heat_equation_kernel_basic"]="basic"
kernels["heat_equation_kernel_fma"]="fma"
kernels["heat_equation_kernel_loop_unroll"]="loop_unroll"

# Loop over block sizes in x and y ensuring x * y <= 1024 for sm 7.5/7.2
for x in 1 2 4 8 16 32 64 128 256 512 1024; do
    for y in 1 2 4 8 16 32 64 128 256 512 1024; do
        if [ $((x * y)) -le 1024 ]; then
            for kernel_name in "${!kernels[@]}"; do
                solver_type=${kernels[$kernel_name]}
                echo "Running ncu with kernel=$kernel_name, solver=$solver_type, block sizes x=$x, y=$y" | tee -a "$output_file"
                ncu --kernel-name "$kernel_name" --launch-skip 32 --launch-count 1 "./bin/heat_equation_solver" gpu "$solver_type" "$x" "$y" >> "$output_file" 2>&1
                echo "---------------------------------" | tee -a "$output_file"
            done
        fi
    done
done
