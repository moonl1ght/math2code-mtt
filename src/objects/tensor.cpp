#include "tensor.h"

std::unique_ptr<Tensor> Tensor::create_random(std::vector<int> shape, float min, float max, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min, max);
    auto tensor = std::make_unique<Tensor>(shape);
    for (int i = 0; i < tensor->_total_size; i++) {
        tensor->_data[i] = dist(gen);
    }
    return tensor;
}

float Tensor::get(std::vector<int> indices) const {
    if (!check_indices(indices)) {
        throw std::runtime_error("Invalid indices");
    }
    return _data[calculate_flat_index(indices)];
}

void Tensor::set(std::vector<int> indices, float value) {
    if (!check_indices(indices)) {
        throw std::runtime_error("Invalid indices");
    }
    _data[calculate_flat_index(indices)] = value;
}

void Tensor::print(bool inline_mode) {
    if (inline_mode) {
        for (int i = 0; i < _total_size; i++) {
            std::cout << " " << std::fixed << std::setprecision(2) << _data[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "[";
        for (int i = 0; i < _total_size; i++) {
            int index = 1;
            for (std::vector<int>::size_type j = 1; j < _shape.size(); j++) {
                index *= _shape[_shape.size() - j];
                if (i % index == 0) {
                    std::cout << "[";
                }
            }
            std::cout << " " << std::fixed << std::setprecision(2) << _data[i] << " ";
            index = 1;
            bool was_closed = false;
            for (std::vector<int>::size_type j = 1; j < _shape.size(); j++) {
                index *= _shape[_shape.size() - j];
                if ((i + 1) % index == 0) {
                    std::cout << "]";
                    was_closed = true;
                }
            }
            if (was_closed && i != _total_size - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

void Tensor::squeeze() {
    std::erase(_shape, 1);
    calculate_strides();
}

void Tensor::unsqueeze(int index) {
    _shape.insert(_shape.begin() + index, 1);
    calculate_strides();
}

/// MARK: - Private Methods

void Tensor::calculate_strides() {
    _strides = std::vector<int>(_shape.size(), 1);
    for (int i = _strides.size() - 2; i >= 0; i--) {
        _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
    std::cout << std::endl;
}

int Tensor::calculate_flat_index(std::vector<int> indices) const {
    int index = 0;
    for (std::vector<int>::size_type i = 0; i < indices.size() - 1; i++) {
        index += indices[i] * _strides[i];
    }
    index += indices[indices.size() - 1];
    return index;
}

bool Tensor::check_indices(std::vector<int> indices) const {
    if (indices.size() != _shape.size()) {
        return false;
    }
    for (std::vector<int>::size_type i = 0; i < indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= _shape[i]) {
            return false;
        }
    }
    return true;
}

bool Tensor::is_contiguous() const {
    int expected_stride = 1;
    for (int i = _shape.size() - 1; i >= 0; --i) {
        if (_shape[i] > 1 && _strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= _shape[i];
    }
    return true;
}