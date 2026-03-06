#include "matmul.cuh"
#include <cuda/cmath>
#include <stdio.h>

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

#define TILE_SIZE 32

__global__ void tiled_matmul_broadcast_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K,
    int N,
    int num_batch_dims,
    const int* __restrict__ out_batch_shape,
    const int* __restrict__ a_batch_strides, // Pre-computed (0 if broadcasted)
    const int* __restrict__ b_batch_strides, // Pre-computed (0 if broadcasted)
    const int* __restrict__ c_batch_strides
) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int temp_idx = blockIdx.z;
    int offsetA = 0, offsetB = 0, offsetC = 0;

     for (int i = num_batch_dims - 1; i >= 0; --i) {
         int coord = temp_idx % out_batch_shape[i];
         temp_idx /= out_batch_shape[i];
         offsetA += coord * a_batch_strides[i];
         offsetB += coord * b_batch_strides[i];
         offsetC += coord * c_batch_strides[i];
    }

    // Row/Col in the output matrix C
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over the tiles of the input matrices
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        // Load tiles into shared memory (with boundary checks)
        if (row < M && (t * TILE_SIZE + tx) < K)
            tileA[ty][tx] = A[offsetA + row * K + t * TILE_SIZE + tx];
        else
            tileA[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < K)
            tileB[ty][tx] = B[offsetB + (t * TILE_SIZE + ty) * N + col];
        else
            tileB[ty][tx] = 0.0f;

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute the product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    if (row < M && col < N) {
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

void launchTiledMatmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K,
    int N,
    int batch_size,
    int num_batch_dims,
    const int* __restrict__ out_batch_shape,
    const int* __restrict__ a_batch_strides,
    const int* __restrict__ b_batch_strides,
    const int* __restrict__ c_batch_strides
) {
    dim3 blockDim(ThreadsPerBlock_x, ThreadsPerBlock_y);
    dim3 gridDim(
        cuda::ceil_div(M, ThreadsPerBlock_x),
        cuda::ceil_div(N, ThreadsPerBlock_y),
        batch_size
    );
    tiled_matmul_broadcast_kernel<<<gridDim, blockDim>>>(
        A, B, C,
        M, K, N,
        num_batch_dims, out_batch_shape, a_batch_strides, b_batch_strides, c_batch_strides
    );
}