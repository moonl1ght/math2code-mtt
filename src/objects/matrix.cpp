#include "matrix.h"

std::unique_ptr<Matrix> Matrix::create_random(
    int rows, int cols, float min, float max, unsigned int seed
) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min, max);

    auto matrix = std::make_unique<Matrix>(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->set(i, j, dist(gen));
        }
    }
    return matrix;
}

std::unique_ptr<Matrix> Matrix::transpose_cpu() const {
    auto transposed = std::make_unique<Matrix>(_cols, _rows);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            transposed->set(j, i, get(i, j));
        }
    }
    return transposed;
}

std::unique_ptr<Matrix> Matrix::transpose_gpu() const {
    if (_gpu_data == nullptr) {
        throw std::runtime_error("GPU memory not allocated");
    }
    auto transposed = std::make_unique<Matrix>(_cols, _rows);
    transposed->allocate_gpu_memory();
    launchTranspose(_rows, _cols, transposed->_gpu_data, _gpu_data);
    transposed->copy_to_host();
    return transposed;
}

std::unique_ptr<Matrix> Matrix::transpose_naive_gpu() const {
    if (_gpu_data == nullptr) {
        std::runtime_error("GPU memory not allocated");
    }
    auto transposed = std::make_unique<Matrix>(_cols, _rows);
    transposed->allocate_gpu_memory();
    launchNaiveTranspose(_rows, _cols, transposed->_gpu_data, _gpu_data);
    transposed->copy_to_host();
    return transposed;
}