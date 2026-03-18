#ifndef RELU_BACKWARD_CUH
#define RELU_BACKWARD_CUH

void launchReluBackwardInplace(float* __restrict__ grad, float* __restrict__ output, int size);

#endif