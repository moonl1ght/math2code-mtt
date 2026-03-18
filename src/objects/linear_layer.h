#pragma once

#include <iostream>
#include "tensor.h"
#include "../operations/MatmulOperation.h"
#include "../operations/TensorOperation.h"
#include "../utils/utils.h"


class LinearLayer {
public:
    std::string name;
    std::unique_ptr<Tensor> weights; // Shape: (in_features, out_features)
    std::unique_ptr<Tensor> bias; // Shape: (1, out_features)
    std::unique_ptr<Tensor> output; // Shape: (batch_size, out_features)
    std::unique_ptr<Tensor> grad_weights; // Shape: (in_features, out_features)
    std::unique_ptr<Tensor> grad_bias; // Shape: (out_features)
    std::unique_ptr<Tensor> grad_input; // Shape: (batch_size, in_features)

    LinearLayer(int in_features, int out_features, int batch_size, std::string name): name(name) {
        // in PyTorch, it's (out_features, in_features),
        // but for now we use (in_features, out_features) for simplicity
        float scale = std::sqrt(2.0f / in_features);
        weights = Tensor::create_random({in_features, out_features}, -scale, scale);
        bias = Tensor::create_zeros({1, out_features});
        output = Tensor::create_zeros({batch_size, out_features});
        grad_weights = Tensor::create_zeros({in_features, out_features});
        grad_bias = Tensor::create_zeros({1, out_features});
        grad_input = Tensor::create_zeros({batch_size, in_features});
    }

    void forward(Tensor& input) {
        MatmulOperation::matmul_inplace(input, *weights, *output);
        TensorOperation::add_inplace(*output, *bias);
    }

    void prepare_for_gpu_work() {
        weights->prepare_for_gpu_work();
        bias->prepare_for_gpu_work();
        output->prepare_for_gpu_work();
        grad_weights->prepare_for_gpu_work();
        grad_bias->prepare_for_gpu_work();
        grad_input->prepare_for_gpu_work();
    }

    void backward(Tensor& input, Tensor& grad, float learning_rate);
};