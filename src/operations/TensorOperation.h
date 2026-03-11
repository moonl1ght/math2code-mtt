#pragma once

#include <iostream>
#include "../objects/tensor.h"

namespace TensorOperation {
    void add_inplace(Tensor& A, const Tensor& B);
    void relu_inplace(Tensor& A);
    void softmax_inplace_cpu(Tensor& A);
}