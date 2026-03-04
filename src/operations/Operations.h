#pragma once

#include <iostream>
#include <memory>
#include "../../src_cuda/matmul.cuh"
#include "../objects/matrix.h"
#include "../objects/tensor.h"

namespace Operations {
    std::unique_ptr<Tensor> naive_matmul_nd(Tensor& A, Tensor& B);
    std::unique_ptr<Tensor> matmul_nd(Tensor& A, Tensor& B);
}