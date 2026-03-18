#include "sumrow.cuh"
#include <cuda/cmath>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void sumrow_kernel(const float* __restrict__ A, float* __restrict__ B, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        B[idx] = 0.0f;
    }

    for (int i = 0; i < batch_size; i ++) {
        if (idx < size) {
            B[idx] += A[idx + i * size];
        }
    }
}

void launchSumRowsInplace(const float* __restrict__ A, float* __restrict__ B, int size, int batch_size) {
    int blocksPerGrid = cuda::ceil_div(size, THREADS_PER_BLOCK);
    sumrow_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(A, B, size, batch_size);
}