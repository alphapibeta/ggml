#include "MatrixMul.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

void printMatrix(const float* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <size> <block-x> <block-y>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int blockX = std::atoi(argv[2]);
    int blockY = std::atoi(argv[3]);

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);

    // CPU Matrix Multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(A.data(), B.data(), C.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Matrix Multiplication Result:\n";
    // printMatrix(C.data(), N);
    std::cout << "Time taken: " << std::chrono::duration<double>(end - start).count() << " seconds.\n";



    

    // GPU Matrix Multiplication
    start = std::chrono::high_resolution_clock::now();
    matrixMulGPU(A.data(), B.data(), C.data(), N, blockX, blockY);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Matrix Multiplication Result:\n";
    // printMatrix(C.data(), N);
    std::cout << "Time taken: " << std::chrono::duration<double>(end - start).count() << " seconds.\n";

    return 0;
}
