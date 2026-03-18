#pragma once

#include "linear_layer.h"
#include "tensor.h"
#include "../operations/TensorOperation.h"

#include <algorithm>

class MNISTModel {
public:
    LinearLayer fc1; // 784 -> 128
    LinearLayer fc2; // 128 -> 10

    MNISTModel(int batch_size) : fc1(784, 128, batch_size, "fc1"), fc2(128, 10, batch_size, "fc2") {}

    // for debugging
    MNISTModel(
        int input_size, int hidden_size, int output_size, int batch_size
    ) : fc1(input_size, hidden_size, batch_size, "fc1"), fc2(hidden_size, output_size, batch_size, "fc2") {}

    void prepare_for_gpu_work() {
        fc1.prepare_for_gpu_work();
        fc2.prepare_for_gpu_work();
    }

    void forward(Tensor& input) {
        fc1.forward(input);
        TensorOperation::relu_inplace(*fc1.output);
        fc1.output->copy_to_host(); // save relu output for backward pass
        fc2.forward(*fc1.output);
        cudaDeviceSynchronize();
        fc2.output->copy_to_host();
        TensorOperation::softmax_inplace_cpu(*fc2.output);
        // std::cout << "fc2.output: " << std::endl;
        // fc2.output->print();
    }

    // Compute gradients using combined softmax + cross-entropy backward.
    // batch_labels must point to batch_size labels matching the input batch.
    void backward(Tensor& input, std::vector<int>& targets, float lr);

    // Training step
    float sgd_step(Tensor& input, std::vector<int>& targets, float lr);

    std::vector<int> predict(Tensor& input);
};