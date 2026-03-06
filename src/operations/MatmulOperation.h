#pragma once

#include <iostream>
#include <memory>
#include "../../src_cuda/matmul.cuh"
#include "../objects/matrix.h"
#include "../objects/tensor.h"

namespace MatmulOperation {
    struct MatmulInfo {
        int M;
        int K_A;
        int K_B;
        int N;
        int batch_size;
        int num_batch_dims;
        std::vector<int> output_shape;
    };

    MatmulInfo get_matmul_info(Tensor& A, Tensor& B);
    void matmul(Tensor& A, Tensor& B, Tensor& C, MatmulInfo& matmul_info);

    std::unique_ptr<Tensor> naive_matmul_nd(Tensor& A, Tensor& B);
    std::unique_ptr<Tensor> matmul_nd(Tensor& A, Tensor& B);
    std::unique_ptr<Tensor> matmul_cpu(Tensor& A, Tensor& B);

    void matmul_inplace(Tensor& A, Tensor& B, Tensor& C);
}