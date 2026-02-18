#include "lesson_2_1.h"
#include "../../src/objects/matrix.h"
#include <iostream>
#include <cuda_runtime.h>

void matrix_transpose(bool is_naive = false);

void lesson_2_1() {
    std::cout << "Lesson 2.1" << std::endl;
    matrix_transpose(false);
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
    } else {
        transposed = matrix->transpose_gpu();
    }
    std::cout << "\nTransposed Matrix:" << std::endl;
    transposed->print();
}