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

    Tensor(std::vector<int> shape): _shape(shape) {
        _total_size = std::accumulate(
            shape.begin(), shape.end(), 1, std::multiplies<int>()
        );
        calculate_strides();
        _data = new float[_total_size];
    };

    ~Tensor() {
        delete[] _data;
    };

    void set(std::vector<int> indices, float value);
    float get(std::vector<int> indices) const;
    void squeeze(); // remove the dimension of size 1
    void unsqueeze(int index); // add the dimension of size 1

    std::vector<int> shape() const { return _shape; }
    std::vector<int> strides() const { return _strides; }
    int total_size() const { return _total_size; }

    void print(bool inline_mode = false);
    void allocate_gpu_memory() { };
    void copy_to_gpu() { };
    void copy_to_host() { };

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