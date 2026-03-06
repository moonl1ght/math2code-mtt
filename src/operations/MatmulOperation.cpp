#include "MatmulOperation.h"

#include "../utils/utils.h"

MatmulOperation::MatmulInfo MatmulOperation::get_matmul_info(Tensor& A, Tensor& B) {
    if (A.shape().size() == 1 && B.shape().size() == 1) {
        A.unsqueeze(0);
        B.unsqueeze(0);
    }
    else if (A.shape().size() < B.shape().size()) {
        A.align_broadcast_to_higher_dimensions(B);
    }
    else if (A.shape().size() > B.shape().size()) {
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

    int num_batch_dims = A.shape().size() - 2;
    return { M, K_A, K_B, N, batch_size, num_batch_dims, output_shape };
}

std::unique_ptr<Tensor> MatmulOperation::naive_matmul_nd(Tensor& A, Tensor& B) {
    if (A.shape().size() == 1 && B.shape().size() == 1) {
        A.unsqueeze(0);
        B.unsqueeze(0);
    }
    else if (A.shape().size() < B.shape().size()) {
        A.align_broadcast_to_higher_dimensions(B);
    }
    else if (A.shape().size() > B.shape().size()) {
        B.align_broadcast_to_higher_dimensions(A);
    }
    int M = A.shape().back() - 1;
    int K_A = A.shape().back();
    int K_B = B.shape().back() - 1;
    int N = B.shape().back();
    if (K_A != K_B) {
        throw std::runtime_error("Last two dimensions of A and B do not match");
    }
    std::vector<int> output_shape = { A.batch_size(), M, N };
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

void MatmulOperation::matmul(Tensor& A, Tensor& B, Tensor& C, MatmulOperation::MatmulInfo& matmul_info) {
    int* a_batch_strides;
    int* a_batch_strides_dev;
    cudaMallocHost(&a_batch_strides, matmul_info.num_batch_dims * sizeof(int));
    cudaMalloc(&a_batch_strides_dev, matmul_info.num_batch_dims * sizeof(int));
    int* b_batch_strides;
    int* b_batch_strides_dev;
    cudaMallocHost(&b_batch_strides, matmul_info.num_batch_dims * sizeof(int));
    cudaMalloc(&b_batch_strides_dev, matmul_info.num_batch_dims * sizeof(int));
    int* c_batch_strides;
    int* c_batch_strides_dev;
    cudaMallocHost(&c_batch_strides, matmul_info.num_batch_dims * sizeof(int));
    cudaMalloc(&c_batch_strides_dev, matmul_info.num_batch_dims * sizeof(int));
    for (int i = 0; i < matmul_info.num_batch_dims; i++) {
        a_batch_strides[i] = A.shape()[i] == 1 ? 0 : A.strides()[i];
        b_batch_strides[i] = B.shape()[i] == 1 ? 0 : B.strides()[i];
        c_batch_strides[i] = C.strides()[i];
    }
    cudaMemcpy(a_batch_strides_dev, a_batch_strides, matmul_info.num_batch_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_batch_strides_dev, b_batch_strides, matmul_info.num_batch_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_batch_strides_dev, c_batch_strides, matmul_info.num_batch_dims * sizeof(int), cudaMemcpyHostToDevice);

    int* out_batch_shape;
    int* out_batch_shape_dev;
    cudaMallocHost(&out_batch_shape, matmul_info.num_batch_dims * sizeof(int));
    cudaMalloc(&out_batch_shape_dev, matmul_info.num_batch_dims * sizeof(int));
    for (int i = 0; i < matmul_info.num_batch_dims; i++) {
        out_batch_shape[i] = matmul_info.output_shape[i];
    }
    cudaMemcpy(out_batch_shape_dev, out_batch_shape, matmul_info.num_batch_dims * sizeof(int), cudaMemcpyHostToDevice);

    launchTiledMatmul(
        A.gpu_data(), B.gpu_data(), C.gpu_data(),
        matmul_info.M, matmul_info.K_A, matmul_info.N,
        matmul_info.batch_size, matmul_info.num_batch_dims, out_batch_shape_dev,
        a_batch_strides_dev, b_batch_strides_dev, c_batch_strides_dev
    );
    cudaDeviceSynchronize();
    cudaFreeHost(a_batch_strides);
    cudaFreeHost(b_batch_strides);
    cudaFreeHost(c_batch_strides);
    cudaFreeHost(out_batch_shape);
    cudaFree(a_batch_strides_dev);
    cudaFree(b_batch_strides_dev);
    cudaFree(c_batch_strides_dev);
    cudaFree(out_batch_shape_dev);
}

void MatmulOperation::matmul_inplace(Tensor& A, Tensor& B, Tensor& C) {
    auto matmul_info = get_matmul_info(A, B);
    matmul(A, B, C, matmul_info);
}

std::unique_ptr<Tensor> MatmulOperation::matmul_nd(Tensor& A, Tensor& B) {

    auto matmul_info = get_matmul_info(A, B);
    auto C = std::make_unique<Tensor>(matmul_info.output_shape);
    C->allocate_gpu_memory();
    matmul(A, B, *C, matmul_info);
    return C;
}

std::unique_ptr<Tensor> MatmulOperation::matmul_cpu(Tensor& A, Tensor& B) {
    if (A.shape().size() == 1 && B.shape().size() == 1) {
        A.unsqueeze(0);
        B.unsqueeze(0);
    }
    else if (A.shape().size() < B.shape().size()) {
        A.align_broadcast_to_higher_dimensions(B);
    }
    else if (A.shape().size() > B.shape().size()) {
        B.align_broadcast_to_higher_dimensions(A);
    }

    int M = *(A.shape().end() - 2);
    int K_A = A.shape().back();
    int K_B = *(B.shape().end() - 2);
    int N = B.shape().back();
    if (K_A != K_B) {
        throw std::runtime_error("Inner dimensions of A and B do not match");
    }
    int K = K_A;

    int num_batch_dims = (int)A.shape().size() - 2;
    int batch_size = 1;
    std::vector<int> output_shape;
    for (int i = 0; i < num_batch_dims; i++) {
        if (A.shape()[i] != B.shape()[i]) {
            if (A.shape()[i] != 1 && B.shape()[i] != 1) {
                throw std::runtime_error("Batch dimensions of A and B do not match");
            }
        }
        int max_dim = std::max(A.shape()[i], B.shape()[i]);
        batch_size *= max_dim;
        output_shape.push_back(max_dim);
    }
    output_shape.push_back(M);
    output_shape.push_back(N);

    auto C = std::make_unique<Tensor>(output_shape);

    // Batch strides with broadcast: stride 0 for size-1 dims
    std::vector<int> a_batch_strides(num_batch_dims);
    std::vector<int> b_batch_strides(num_batch_dims);
    std::vector<int> c_batch_strides(num_batch_dims);
    for (int i = 0; i < num_batch_dims; i++) {
        a_batch_strides[i] = A.shape()[i] == 1 ? 0 : A.strides()[i];
        b_batch_strides[i] = B.shape()[i] == 1 ? 0 : B.strides()[i];
        c_batch_strides[i] = C->strides()[i];
    }

    int a_row_stride = A.strides()[A.shape().size() - 2];
    int b_row_stride = B.strides()[B.shape().size() - 2];
    int c_row_stride = C->strides()[C->shape().size() - 2];

    float* a = A.data();
    float* b = B.data();
    float* c = C->data();

    for (int batch = 0; batch < batch_size; batch++) {
        // Decompose flat batch index into per-dim offsets
        int a_offset = 0, b_offset = 0, c_offset = 0;
        int tmp = batch;
        for (int d = num_batch_dims - 1; d >= 0; d--) {
            int idx = tmp % output_shape[d];
            tmp /= output_shape[d];
            a_offset += idx * a_batch_strides[d];
            b_offset += idx * b_batch_strides[d];
            c_offset += idx * c_batch_strides[d];
        }

        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a[a_offset + m * a_row_stride + k] *
                        b[b_offset + k * b_row_stride + n];
                }
                c[c_offset + m * c_row_stride + n] = sum;
            }
        }
    }

    return C;
}