#include "relu.cuh"
#include <cuda/cmath>
#include <stdio.h>

__global__ void relu_kernel(float* __restrict__ A, int size) {
    // we don't use grid stride loop for simplicity
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = max(A[idx], 0.0f);
    }
}

void launchReluInplace(float* __restrict__ A, int size, int threadsPerBlock) {
    int blocksPerGrid = cuda::ceil_div(size, threadsPerBlock);
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, size);
}