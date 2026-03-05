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

void launchTiledMatmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K,
    int N,
    int batch_size,
    int num_batch_dims,
    const int* __restrict__ out_batch_shape,
    const int* __restrict__ a_batch_strides,
    const int* __restrict__ b_batch_strides,
    const int* __restrict__ c_batch_strides
);

#endif