#pragma once

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "../../src_cuda/transpose.cuh"

// strore data in row-major order
class Matrix {
public:

    static std::unique_ptr<Matrix> create_random(int rows, int cols, float min = 0.0f, float max = 1.0f, unsigned int seed = 0);

    Matrix(int rows, int cols)
        : _rows(rows)
        , _cols(cols)
        , _data(new float[rows * cols]) {
    }

    ~Matrix() {
        delete[] _data;
        free_gpu_memory();
    }

    // no copy constructor and assignment operator
    Matrix(const Matrix& other) = delete;
    Matrix& operator=(const Matrix& other) = delete;

    Matrix(Matrix&& other)
        : _rows(other._rows)
        , _cols(other._cols)
        , _data(other._data)
        , _gpu_data(other._gpu_data) {
        other._data = nullptr;
        other._gpu_data = nullptr;
        other._rows = 0;
        other._cols = 0;
    }

    Matrix& operator=(Matrix&& other) {
        if (this == &other) {
            return *this;
        }
        delete[] _data;
        free_gpu_memory();
        _rows = other._rows;
        _cols = other._cols;
        _data = other._data;
        _gpu_data = other._gpu_data;
        other._data = nullptr;
        other._gpu_data = nullptr;
        other._rows = 0;
        other._cols = 0;
        return *this;
    }

    void set(int row, int col, float value) {
        _data[row * _cols + col] = value;
    }

    float get(int row, int col) const {
        return _data[row * _cols + col];
    }

    int rows() const {
        return _rows;
    }

    int cols() const {
        return _cols;
    }

    void print() const {
        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < _cols; j++) {
                std::cout << " " << std::fixed << std::setprecision(2) << _data[i * _cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void allocate_gpu_memory() {
        if (_gpu_data != nullptr) {
            return;
        }
        cudaMalloc(&_gpu_data, _rows * _cols * sizeof(float));
    }

    void free_gpu_memory() {
        if (_gpu_data == nullptr) {
            return;
        }
        cudaFree(_gpu_data);
        _gpu_data = nullptr;
    }

    void copy_to_gpu() {
        if (_gpu_data == nullptr) {
            allocate_gpu_memory();
        }
        cudaMemcpy(
            _gpu_data, _data, _rows * _cols * sizeof(float), cudaMemcpyHostToDevice
        );
    }

    void copy_to_host() {
        if (_gpu_data == nullptr) {
            throw std::runtime_error("GPU memory not allocated");
        }
        cudaMemcpy(
            _data, _gpu_data, _rows * _cols * sizeof(float), cudaMemcpyDeviceToHost
        );
    }

    std::unique_ptr<Matrix> transpose_cpu() const;
    std::unique_ptr<Matrix> transpose_gpu() const;
    std::unique_ptr<Matrix> transpose_naive_gpu() const;
    std::unique_ptr<Matrix> transpose_gpu(float& gpu_time) const;
    std::unique_ptr<Matrix> transpose_naive_gpu(float& gpu_time) const;

    bool operator==(const Matrix& other) const {
        if (_rows != other._rows || _cols != other._cols) {
            return false;
        }
        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < _cols; j++) {
                if (get(i, j) != other.get(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    int _rows;
    int _cols;
    float* _data;
    float* _gpu_data = nullptr;
};