#include "add_weighted.cuh"
#include <cuda/cmath>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void add_weighted_kernel(
    float* __restrict__ A,
    const float* __restrict__ B,
    int size,
    const float learning_rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // New = Old + (-LearningRate * Gradient)
        A[i] = A[i] + (B[i] * -learning_rate);
    }
}

void launchAddWeightedInplace(
    float* __restrict__ A,
    const float* __restrict__ B,
    int size,
    const float learning_rate
) {
    int blocksPerGrid = cuda::ceil_div(size, THREADS_PER_BLOCK);
    add_weighted_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(A, B, size, learning_rate);
}