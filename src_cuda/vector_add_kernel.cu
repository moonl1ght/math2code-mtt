#include "vector_add_kernel.cuh"
#include <cuda/cmath>

// CUDA Kernel Implementation
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Wrapper function to launch kernel (callable from CPU code)
void launchVectorAdd(const float* d_A, const float* d_B, float* d_C,
                     int numElements, int threadsPerBlock) {
    // int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = cuda::ceil_div(numElements, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
}
