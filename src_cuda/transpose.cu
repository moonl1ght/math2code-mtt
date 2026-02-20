#include "transpose.cuh"
#include <cuda/cmath>
#include <iostream>

#define ThreadsPerBlock_x 32
#define ThreadsPerBlock_y 32

__global__ void naiveTranspose(
    int m,
    int n,
    float* output,
    const float* __restrict__ input
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < m && row < n) {
        output[row * m + col] = input[col * n + row];
    }
}

__global__ void transpose(
    int m,
    int n,
    float* output,
    const float* input
) {
    __shared__ float block_data[ThreadsPerBlock_x][ThreadsPerBlock_y + 1]; // +1 to avoid bank conflicts
    // get the row and column of the thread, for boundary check
    const int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    const int myRow = blockDim.y * blockIdx.y + threadIdx.y;

    // get the starting row and column of the block (tile)
    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if( myRow < m && myCol < n ) {
        const int thread_y_idx = tileY + threadIdx.y;
        const int thread_x_idx = tileX + threadIdx.x;
        const int stride_y = blockDim.y * gridDim.y;
        const int stride_x = blockDim.x * gridDim.x;
        for (int i = thread_y_idx; i < m; i += stride_y) {
            for (int j = thread_x_idx; j < n; j += stride_x) {
                block_data[threadIdx.x][threadIdx.y] = input[i * n + j];
            }
        }
        // block_data[threadIdx.x][threadIdx.y] = input[thread_y_idx * n + thread_x_idx];
    }

    // wait till all threads in the block have loaded their data into shared memory
    __syncthreads();

    const int myColSave = blockDim.x * blockIdx.x + threadIdx.y;
    const int myRowSave = blockDim.y * blockIdx.y + threadIdx.x;
    if( myRowSave < m && myColSave < n ) {
        const int thread_y_idx = tileY + threadIdx.x;
        const int thread_x_idx = tileX + threadIdx.y;
        const int stride_y = blockDim.y * gridDim.y;
        const int stride_x = blockDim.x * gridDim.x;
        for (int i = thread_x_idx; i < m; i += stride_x) {
            for (int j = thread_y_idx; j < n; j += stride_y) {
                const int transposed_index = i * m + j;
                output[transposed_index] = block_data[threadIdx.y][threadIdx.x];
            }
        }
    }
}

void launchNaiveTranspose(
    int m,
    int n,
    float* output,
    const float* __restrict__ input
) {
    dim3 blockDim(ThreadsPerBlock_x, ThreadsPerBlock_y);
    dim3 gridDim(cuda::ceil_div(m, ThreadsPerBlock_x), cuda::ceil_div(n, ThreadsPerBlock_y));
    naiveTranspose<<<gridDim, blockDim>>>(m, n, output, input);
}

void launchTranspose(
    int m,
    int n,
    float* output,
    const float* __restrict__ input
) {
    dim3 blockDim(ThreadsPerBlock_x, ThreadsPerBlock_y);
    dim3 gridDim(cuda::ceil_div(n, ThreadsPerBlock_x), cuda::ceil_div(m, ThreadsPerBlock_y));
    transpose<<<gridDim, blockDim>>>(m, n, output, input);
}
