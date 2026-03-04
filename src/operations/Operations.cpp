#include "Operations.h"

std::unique_ptr<Tensor> Operations::naive_matmul_nd(Tensor& A, Tensor& B) {
    auto [M, K_A] = A.last_two_dimensions();
    auto [K_B, N] = B.last_two_dimensions();
    if (K_A != K_B) {
        throw std::runtime_error("Last two dimensions of A and B do not match");
    }
    std::vector<int> output_shape = {A.batch_size(), M, N};
    auto C = std::make_unique<Tensor>(output_shape);
    C->allocate_gpu_memory();
    A.allocate_gpu_memory();
    B.allocate_gpu_memory();
    A.copy_to_gpu();
    B.copy_to_gpu();
    launchNaiveMatmulNd(A.gpu_data(), B.gpu_data(), C->gpu_data(), M, K_A, N, A.batch_size());
    C->copy_to_host();
    return C;
}

std::unique_ptr<Tensor> Operations::matmul_nd(Tensor& A, Tensor& B) {
    std::vector<int> output_shape = {1};
    return std::make_unique<Tensor>(output_shape);
}