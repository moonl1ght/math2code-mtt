#pragma once

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>
#include <cuda_runtime.h>

class Tensor {
public:

    static std::unique_ptr<Tensor> create_random(
        std::vector<int> shape, 
        float min = 0.0f, 
        float max = 1.0f,
        unsigned int seed = 0
    );

    static std::unique_ptr<Tensor> create_zeros(std::vector<int> shape) {
        auto tensor = std::make_unique<Tensor>(shape);
        for (int i = 0; i < tensor->_total_size; i++) {
            tensor->_data[i] = 0.0f;
        }
        return tensor;
    }

    static std::unique_ptr<Tensor> create_ones(std::vector<int> shape) {
        auto tensor = std::make_unique<Tensor>(shape);
        for (int i = 0; i < tensor->_total_size; i++) {
            tensor->_data[i] = 1.0f;
        }
        return tensor;
    }

    Tensor(std::vector<int> shape): _shape(shape) {
        _total_size = std::accumulate(
            shape.begin(), shape.end(), 1, std::multiplies<int>()
        );
        calculate_strides();
        // Allocate pinned memory
        cudaMallocHost((void**)&_data, _total_size * sizeof(float));
    };

    Tensor(std::vector<int> shape, std::vector<float> data): _shape(shape) {
        _total_size = std::accumulate(
            shape.begin(), shape.end(), 1, std::multiplies<int>()
        );
        if (data.size() != _total_size) {
            throw std::runtime_error("Data size does not match shape");
        }
        calculate_strides();
        // Allocate pinned memory
        cudaMallocHost((void**)&_data, _total_size * sizeof(float));
        for (int i = 0; i < _total_size; i++) {
            _data[i] = data[i];
        }
    };

    ~Tensor() {
        cudaFreeHost(_data);
        free_gpu_memory();
    };

    Tensor(Tensor&& other) noexcept
        : _shape(other._shape)
        , _strides(other._strides)
        , _total_size(other._total_size)
        , _data(other._data)
        , _gpu_data(other._gpu_data) {
        other._shape = std::vector<int>();
        other._strides = std::vector<int>();
        other._total_size = 0;
        other._data = nullptr;
        other._gpu_data = nullptr;
    }
    
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        _shape = other._shape;
        _strides = other._strides;
        _total_size = other._total_size;
        _data = other._data;
        _gpu_data = other._gpu_data;
        other._shape = std::vector<int>();
        other._strides = std::vector<int>();
        other._total_size = 0;
        other._data = nullptr;
        other._gpu_data = nullptr;
        return *this;
    }

    Tensor(const Tensor& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;

    float* data() const { return _data; }
    float* gpu_data() const { return _gpu_data; }
    bool is_gpu_allocated() const { return _gpu_data != nullptr; }

    void set(std::vector<int> indices, float value);
    float get(std::vector<int> indices) const;
    void squeeze(); // remove the dimension of size 1
    void unsqueeze(int index); // add the dimension of size 1

    void align_broadcast_to_higher_dimensions(const Tensor& other) {
        if (other.shape().size() < shape().size()) {
            throw std::runtime_error("Other tensor has less dimensions than this tensor");
        }
        while (shape().size() < other.shape().size()) {
            this->unsqueeze(0);
        }
    }

    std::vector<int> shape() const { return _shape; }
    std::vector<int> strides() const { return _strides; }
    int total_size() const { return _total_size; }

    void print(bool inline_mode = false);

    int batch_size() const { 
        if (_shape.size() < 2) {
            return 1;
        }
        return std::accumulate(_shape.begin(), _shape.end() - 2, 1, std::multiplies<int>()); 
    }

    void prepare_for_gpu_work() {
        allocate_gpu_memory();
        copy_to_gpu();
    }

    void allocate_gpu_memory() { 
        if (_gpu_data != nullptr) {
            return;
        }
        cudaMalloc(&_gpu_data, _total_size * sizeof(float));
    };

    void copy_to_gpu() { 
        if (_gpu_data == nullptr) {
            throw std::runtime_error("GPU memory not allocated");
        }
        cudaMemcpy(_gpu_data, _data, _total_size * sizeof(float), cudaMemcpyHostToDevice);
    };

    void copy_to_host() { 
        if (_gpu_data == nullptr) {
            throw std::runtime_error("GPU memory not allocated");
        }
        cudaMemcpy(_data, _gpu_data, _total_size * sizeof(float), cudaMemcpyDeviceToHost);
    };

    bool operator==(const Tensor& other) const {
        return approx_equal(other, 1e-5f);
    }

    bool approx_equal(const Tensor& other, float epsilon = 1e-5f) const {
        if (shape() != other.shape()) {
            return false;
        }
        for (int i = 0; i < total_size(); i++) {
            if (std::abs(data()[i] - other.data()[i]) > epsilon) {
                return false;
            }
        }
        return true;
    }

    void free_gpu_memory() {
        if (_gpu_data == nullptr) {
            return;
        }
        cudaFree(_gpu_data);
        _gpu_data = nullptr;
    }

private:
    std::vector<int> _shape;
    std::vector<int> _strides;
    int _total_size;
    float* _data = nullptr;
    float* _gpu_data = nullptr;

    void calculate_strides();
    bool check_indices(std::vector<int> indices) const;
    int calculate_flat_index(std::vector<int> indices) const;
    bool is_contiguous() const;
};  