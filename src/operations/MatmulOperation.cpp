#include "MatmulOperation.h"

#include "../utils/utils.h"

MatmulOperation::MatmulInfo MatmulOperation::get_matmul_info(
    Tensor& A, Tensor& B, bool transA, bool transB
) {
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
    if (transA) {
        if (M != K_B) {
            throw std::runtime_error("transA Last two dimensions of A and B do not match");
        }
    } else if (transB) {
        if (K_A != N) {
            throw std::runtime_error("transB Last two dimensions of A and B do not match");
        }
    } else {
        if (K_A != K_B) {
            throw std::runtime_error("Last two dimensions of A and B do not match");
        }
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
    output_shape.push_back(transA ? K_A : M);
    output_shape.push_back(transB ? K_B : N);

    int num_batch_dims = A.shape().size() - 2;
    // print_vector("output_shape", output_shape);
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

void MatmulOperation::matmul(
    Tensor& A, Tensor& B, Tensor& C,
    MatmulOperation::MatmulInfo& matmul_info, 
    bool transA, bool transB
) {
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

    int M = transA ? matmul_info.K_A : matmul_info.M;
    int N = transB ? matmul_info.K_B : matmul_info.N;
    int K = transA ? matmul_info.M : matmul_info.K_A;
    launchTiledMatmul(
        A.gpu_data(), B.gpu_data(), C.gpu_data(),
        M, K, N,
        transA, transB,
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

void MatmulOperation::matmul_inplace(
    Tensor& A, Tensor& B, Tensor& C, bool transA, bool transB
) {
    auto matmul_info = get_matmul_info(A, B, transA, transB);
    matmul(A, B, C, matmul_info, transA, transB);
}

std::unique_ptr<Tensor> MatmulOperation::matmul_nd(
    Tensor& A, Tensor& B, bool transA, bool transB
) {
    auto matmul_info = get_matmul_info(A, B, transA, transB);
    auto C = std::make_unique<Tensor>(matmul_info.output_shape);
    C->allocate_gpu_memory();
    matmul(A, B, *C, matmul_info, transA, transB);
    return C;
}

std::unique_ptr<Tensor> MatmulOperation::matmul_cpu(
    Tensor& A, Tensor& B, bool transA, bool transB
) {
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

    // Physical dimensions: A is (rows_A, cols_A), B is (rows_B, cols_B)
    int rows_A = *(A.shape().end() - 2);
    int cols_A = A.shape().back();
    int rows_B = *(B.shape().end() - 2);
    int cols_B = B.shape().back();

    // Effective output dimensions and inner contraction length K:
    //   default:   C(rows_A, cols_B),  K = cols_A = rows_B
    //   transA:    C(cols_A, cols_B),  K = rows_A = rows_B
    //   transB:    C(rows_A, rows_B),  K = cols_A = cols_B
    //   transA+B:  C(cols_A, rows_B),  K = rows_A = cols_B
    int M_out   = transA ? cols_A : rows_A;
    int N_out   = transB ? rows_B : cols_B;
    int K       = transA ? rows_A : cols_A;
    int K_check = transB ? cols_B : rows_B;
    if (K != K_check) {
        throw std::runtime_error("Inner dimensions of A and B do not match");
    }

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
    output_shape.push_back(M_out);
    output_shape.push_back(N_out);

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

    // Physical row strides (stride of the second-to-last dim)
    int a_row_stride = A.strides()[A.shape().size() - 2];
    int b_row_stride = B.strides()[B.shape().size() - 2];
    int c_row_stride = C->strides()[C->shape().size() - 2];

    float* a = A.data();
    float* b = B.data();
    float* c = C->data();

    for (int batch = 0; batch < batch_size; batch++) {
        int a_offset = 0, b_offset = 0, c_offset = 0;
        int tmp = batch;
        for (int d = num_batch_dims - 1; d >= 0; d--) {
            int idx = tmp % output_shape[d];
            tmp /= output_shape[d];
            a_offset += idx * a_batch_strides[d];
            b_offset += idx * b_batch_strides[d];
            c_offset += idx * c_batch_strides[d];
        }

        for (int m = 0; m < M_out; m++) {
            for (int n = 0; n < N_out; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    // transA: read a[k, m] instead of a[m, k]
                    float a_val = transA
                        ? a[a_offset + k * a_row_stride + m]
                        : a[a_offset + m * a_row_stride + k];
                    // transB: read b[n, k] instead of b[k, n]
                    float b_val = transB
                        ? b[b_offset + n * b_row_stride + k]
                        : b[b_offset + k * b_row_stride + n];
                    sum += a_val * b_val;
                }
                c[c_offset + m * c_row_stride + n] = sum;
            }
        }
    }

    return C;
}