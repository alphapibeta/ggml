#include "FiniteDifference.h"

struct ggml_tensor* FiniteDifference::laplacian_2d(struct ggml_context* ctx, struct ggml_tensor* u, float dx, float dy) {
    int nx = u->ne[0];
    int ny = u->ne[1];
    
    struct ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);
    
    float* u_data = (float*)u->data;
    float* result_data = (float*)result->data;
    
    float dx2 = dx * dx;
    float dy2 = dy * dy;
    
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            int idx = i + j * nx;
            result_data[idx] = (u_data[idx-1] + u_data[idx+1] - 2*u_data[idx]) / dx2 +
                               (u_data[idx-nx] + u_data[idx+nx] - 2*u_data[idx]) / dy2;
        }
    }
    
    return result;
}
