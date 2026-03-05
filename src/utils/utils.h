#include <iostream>
#include <vector>
#include <string>

template <typename T>
void print_vector(const std::string& name, const std::vector<T>& vec) {
    std::cout << name << ": [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << " ]" << std::endl;
}

// Overload for raw pointers (useful for CUDA host-side buffers)
template <typename T>
void print_raw_ptr(const std::string& name, const T* ptr, size_t size) {
    std::cout << name << " (raw): [ ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << ptr[i] << (i == size - 1 ? "" : ", ");
    }
    std::cout << " ]" << std::endl;
}