#pragma once
// CPU function declaration
void matrixMulCPU(const float* A, const float* B, float* C, int N);

// GPU function declaration
void matrixMulGPU(const float* A, const float* B, float* C, int N, int blockX, int blockY);
