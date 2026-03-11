#pragma once

#include "linear_layer.h"
#include "tensor.h"
#include "../operations/TensorOperation.h"

#include <algorithm>

class MNISTModel {
public:
    LinearLayer fc1; // 784 -> 128
    LinearLayer fc2; // 128 -> 10

    MNISTModel(int batch_size) : fc1(784, 128, batch_size), fc2(128, 10, batch_size) {}

    void prepare_for_gpu_work() {
        fc1.prepare_for_gpu_work();
        fc2.prepare_for_gpu_work();
    }

    void forward(Tensor& input) {
        fc1.forward(input);
        TensorOperation::relu_inplace(*fc1.output);
        fc2.forward(*fc1.output);
        cudaDeviceSynchronize();
        fc2.output->copy_to_host();
        TensorOperation::softmax_inplace_cpu(*fc2.output);
    }

    std::vector<int> predict(Tensor& input);
};