#pragma once

#include <iostream>
#include "../objects/tensor.h"

namespace TensorOperation {
    void add_inplace(Tensor& A, const Tensor& B);
    void relu_inplace(Tensor& A);
    void softmax_inplace_cpu(Tensor& A);
    float cross_entropy_loss_cpu(const Tensor& predictions, const std::vector<int>& labels);
    void sum_rows_inplace_cpu(const Tensor& A, Tensor& B);
    void sum_rows_inplace(const Tensor& A, Tensor& B);
    void add_weighted_inplace(Tensor& A, const Tensor& B, float learning_rate);
    void relu_backward_inplace(Tensor& grad, const Tensor& output);
}