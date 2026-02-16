#ifndef VECTOR_ADD_KERNEL_CUH
#define VECTOR_ADD_KERNEL_CUH

// CUDA kernel declaration
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements);

// Wrapper function to launch kernel from CPU code
void launchVectorAdd(const float* d_A, const float* d_B, float* d_C,
                     int numElements, int threadsPerBlock);

#endif // VECTOR_ADD_KERNEL_CUH
