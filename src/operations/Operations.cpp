#include "Operations.h"

#include "../utils/utils.h"

std::unique_ptr<Tensor> Operations::naive_matmul_nd(Tensor& A, Tensor& B) {
    if (A.shape().size() == 1 && B.shape().size() == 1) {
        A.unsqueeze(0);
        B.unsqueeze(0);
    } else if (A.shape().size() < B.shape().size()) {
        A.align_broadcast_to_higher_dimensions(B);
    } else if (A.shape().size() > B.shape().size()) {
        B.align_broadcast_to_higher_dimensions(A);
    }
    int M = A.shape().back() - 1;
    int K_A = A.shape().back();
    int K_B = B.shape().back() - 1;
    int N = B.shape().back();
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
    cudaDeviceSynchronize();
    C->copy_to_host();
    return C;
}

std::unique_ptr<Tensor> Operations::matmul_nd(Tensor& A, Tensor& B) {
    if (A.shape().size() == 1 && B.shape().size() == 1) {
        A.unsqueeze(0);
        B.unsqueeze(0);
    } else if (A.shape().size() < B.shape().size()) {
        A.align_broadcast_to_higher_dimensions(B);
    } else if (A.shape().size() > B.shape().size()) {
        B.align_broadcast_to_higher_dimensions(A);
    }
    int M = *(A.shape().end() - 2);
    int K_A = A.shape().back();
    int K_B = *(B.shape().end() - 2);
    int N = B.shape().back();
    if (K_A != K_B) {
        throw std::runtime_error("Last two dimensions of A and B do not match");
    }
    int batch_size = 1;
    std::vector<int> output_shape;
    for (int i = 0; i < A.shape().size() - 2; i++) {
        if (A.shape()[i] != B.shape()[i]) {
            if (A.shape()[i] != 1 && B.shape()[i] != 1) {
                throw std::runtime_error("Dimensions of A and B do not match");
            }
        }
        int max_shape = std::max(A.shape()[i], B.shape()[i]);
        batch_size *= max_shape;
        output_shape.push_back(max_shape);
    }
    output_shape.push_back(M);
    output_shape.push_back(N);

    std::cout << "Output shape: ";
    for (int i = 0; i < output_shape.size(); i++) {
        std::cout << output_shape[i] << " ";
    }
    std::cout << std::endl;

    int num_batch_dims = A.shape().size() - 2;

    auto C = std::make_unique<Tensor>(output_shape);
    
    int* a_batch_strides;
    int* a_batch_strides_dev;
    cudaMallocHost(&a_batch_strides, num_batch_dims * sizeof(int));
    cudaMalloc(&a_batch_strides_dev, num_batch_dims * sizeof(int));
    int* b_batch_strides;
    int* b_batch_strides_dev;
    cudaMallocHost(&b_batch_strides, num_batch_dims * sizeof(int));
    cudaMalloc(&b_batch_strides_dev, num_batch_dims * sizeof(int));
    int* c_batch_strides;
    int* c_batch_strides_dev;
    cudaMallocHost(&c_batch_strides, num_batch_dims * sizeof(int));
    cudaMalloc(&c_batch_strides_dev, num_batch_dims * sizeof(int));
    for (int i = 0; i < num_batch_dims; i++) {
        a_batch_strides[i] = A.shape()[i] == 1 ? 0 : A.strides()[i];
        b_batch_strides[i] = B.shape()[i] == 1 ? 0 : B.strides()[i];
        c_batch_strides[i] = C->strides()[i];
    }
    cudaMemcpy(a_batch_strides_dev, a_batch_strides, num_batch_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_batch_strides_dev, b_batch_strides, num_batch_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_batch_strides_dev, c_batch_strides, num_batch_dims * sizeof(int), cudaMemcpyHostToDevice);

    int* out_batch_shape;
    int* out_batch_shape_dev;
    cudaMallocHost(&out_batch_shape, num_batch_dims * sizeof(int));
    cudaMalloc(&out_batch_shape_dev, num_batch_dims * sizeof(int));
    for (int i = 0; i < num_batch_dims; i++) {
        out_batch_shape[i] = output_shape[i];
    }
    cudaMemcpy(out_batch_shape_dev, out_batch_shape, num_batch_dims * sizeof(int), cudaMemcpyHostToDevice);


    print_raw_ptr("a_batch_strides", a_batch_strides, num_batch_dims);
    print_raw_ptr("b_batch_strides", b_batch_strides, num_batch_dims);
    print_raw_ptr("c_batch_strides", c_batch_strides, num_batch_dims);
    print_raw_ptr("out_batch_shape", out_batch_shape, num_batch_dims);
    C->allocate_gpu_memory();
    A.allocate_gpu_memory();
    B.allocate_gpu_memory();
    A.copy_to_gpu();
    B.copy_to_gpu();
    launchTiledMatmul(
        A.gpu_data(), B.gpu_data(), C->gpu_data(), 
        M, K_A, N, 
        A.batch_size(), num_batch_dims, out_batch_shape, a_batch_strides, b_batch_strides, c_batch_strides
    );
    cudaDeviceSynchronize();
    C->copy_to_host();
    cudaFreeHost(a_batch_strides);
    cudaFreeHost(b_batch_strides);
    cudaFreeHost(c_batch_strides);
    cudaFreeHost(out_batch_shape);
    cudaFree(a_batch_strides_dev);
    cudaFree(b_batch_strides_dev);
    cudaFree(c_batch_strides_dev);
    cudaFree(out_batch_shape_dev);
    return C;
}