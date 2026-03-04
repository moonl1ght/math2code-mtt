#ifndef MATMUL_CUH
#define MATMUL_CUH

void launchNaiveMatmulNd(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M,
    int K,
    int N,
    int batch_size
);

#endif