#ifndef FINITE_DIFFERENCE_H
#define FINITE_DIFFERENCE_H

#include "ggml.h"

class FiniteDifference {
public:
    static struct ggml_tensor* laplacian_2d(struct ggml_context* ctx, struct ggml_tensor* u, float dx, float dy);
    static void laplacian_2d_gpu(struct ggml_tensor* u, struct ggml_tensor* result, float dx, float dy);
};

#endif // FINITE_DIFFERENCE_H
