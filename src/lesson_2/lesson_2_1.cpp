#include "lesson_2_1.h"
#include "../../src/objects/matrix.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

void matrix_transpose(bool is_naive = false);
void benchmark_transpose(int rows, int cols);

void lesson_2_1() {
    std::cout << "Lesson 2.1" << std::endl;
    matrix_transpose(false);
    benchmark_transpose(4096, 4096);
}

void matrix_transpose(bool is_naive) {
    std::cout << "Matrix Transpose (naive: " << is_naive << "):" << std::endl;
    auto matrix = new Matrix(3, 3);
    for (int i = 0; i < matrix->rows(); i++) {
        for (int j = 0; j < matrix->cols(); j++) {
            matrix->set(i, j, i * matrix->cols() + j);
        }
    }
    matrix->print();
    matrix->allocate_gpu_memory();
    matrix->copy_to_gpu();
    std::unique_ptr<Matrix> transposed;
    if (is_naive) {
        transposed = matrix->transpose_naive_gpu();
    }
    else {
        transposed = matrix->transpose_gpu();
    }
    std::cout << "\nTransposed Matrix:" << std::endl;
    transposed->print();
}

void benchmark_transpose(int rows, int cols) {
    std::cout << "\n=== Benchmark: " << rows << "x" << cols
        << " matrix ===" << std::endl;

    auto matrix = Matrix::create_random(rows, cols);
    matrix->allocate_gpu_memory();
    matrix->copy_to_gpu();

    // --- CPU ---
    double cpu_ms = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto resultCPU = matrix->transpose_cpu();
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU (naive) ---
    float naive_ms;
    auto resultNaive = matrix->transpose_naive_gpu(naive_ms);

    // --- GPU (shared memory) ---
    float gpu_ms;
    auto resultGPU = matrix->transpose_gpu(gpu_ms);

    if (*resultCPU == *resultNaive) {
        std::cout << "Naive GPU transpose results match CPU transpose" << std::endl;
    }
    else {
        std::cout << "Naive GPU transpose results mismatch CPU transpose" << std::endl;
    }
    if (*resultCPU == *resultGPU) {
        std::cout << "Shared memory GPU transpose results match CPU transpose" << std::endl;
    }
    else {
        std::cout << "Shared memory GPU transpose results mismatch CPU transpose" << std::endl;
    }

    std::cout << "CPU transpose time: " << cpu_ms << " ms" << std::endl;
    std::cout << "Naive GPU transpose time: " << naive_ms << " ms" << std::endl;
    std::cout << "Shared memory GPU transpose time: " << gpu_ms << " ms" << std::endl;
    std::cout << "Speedup naive GPU vs CPU: " << cpu_ms / naive_ms << "x" << std::endl;
    std::cout << "Speedup shared memory GPU vs naive GPU: " << naive_ms / gpu_ms << "x" << std::endl;
    std::cout << "Speedup shared memory GPU vs CPU: " << cpu_ms / gpu_ms << "x" << std::endl;
}