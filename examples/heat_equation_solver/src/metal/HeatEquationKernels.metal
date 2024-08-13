// HeatEquationKernels.metal
#include <metal_stdlib>
using namespace metal;

kernel void heatEquationKernel(
    device float* u [[buffer(0)]],
    device float* u_next [[buffer(1)]],
    constant int& nx [[buffer(2)]],
    constant int& ny [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant float& dx [[buffer(5)]],
    constant float& dy [[buffer(6)]],
    constant float& dt [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int i = gid.x;
    int j = gid.y;
    
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = i + j * nx;
        float laplacian = (u[idx-1] + u[idx+1] - 2*u[idx]) / (dx*dx) +
                          (u[idx-nx] + u[idx+nx] - 2*u[idx]) / (dy*dy);
        u_next[idx] = u[idx] + alpha * dt * laplacian;
    }
}
