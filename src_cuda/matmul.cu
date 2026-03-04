#include "matmul.cuh"
#include <cuda/cmath>

#define ThreadsPerBlock_x 32
#define ThreadsPerBlock_y 32

__global__ void naive_matmul_nd_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M,
    int K,
    int N
) {
    // Identify which batch (matrix)
    int batch_idx = blockIdx.z;

    // Identify the row and column within that matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for the matrix
    if (row < M && col < N) {
        // Calculate the linear offset to the start of the current slice
        int offsetA = batch_idx * (M * K);
        int offsetB = batch_idx * (K * N);
        int offsetC = batch_idx * (M * N);

        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[offsetA + row * K + k] * B[offsetB + k * N + col];
        }
        C[offsetC + row * N + col] = sum;
    }
}

void launchNaiveMatmulNd(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M,
    int K,
    int N,
    int batch_size
) {
    dim3 blockDim(ThreadsPerBlock_x, ThreadsPerBlock_y);
    dim3 gridDim(cuda::ceil_div(M, ThreadsPerBlock_x), cuda::ceil_div(N, ThreadsPerBlock_y), batch_size);
    naive_matmul_nd_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
}