#ifndef ADD_WEIGHTED_CUH
#define ADD_WEIGHTED_CUH

void launchAddWeightedInplace(
    float* __restrict__ A,
    const float* __restrict__ B,
    int size,
    const float learning_rate
);

#endif