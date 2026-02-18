#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH

void launchNaiveTranspose(
    int m,
    int n,
    float* output,
    const float* __restrict__ input
);

void launchTranspose(
    int m,
    int n,
    float* output,
    const float* __restrict__ input
);

#endif

