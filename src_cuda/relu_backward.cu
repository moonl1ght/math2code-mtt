#include "relu_backward.cuh"
#include <cuda/cmath>

#define THREADS_PER_BLOCK 256

__global__ void relu_backward_kernel(float* __restrict__ grad, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = grad[idx] * (output[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

void launchReluBackwardInplace(float* __restrict__ grad, float* __restrict__ output, int size) {
    int blocksPerGrid = cuda::ceil_div(size, THREADS_PER_BLOCK);
    relu_backward_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(grad, output, size);
}