#ifndef RELU_CUH
#define RELU_CUH

void launchReluInplace(float* __restrict__ A, int size, int threadsPerBlock);

#endif