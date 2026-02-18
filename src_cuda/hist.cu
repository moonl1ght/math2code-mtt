#include "hist.cuh"
#include <cuda/cmath>

__global__ void naiveHist(
    int* bins,
    const int numBins,
    const double bin_size,
    const int min_value,
    const int max_value,
    const int *__restrict__ input,
    const int arraySize
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes elements via grid-stride loop
    // if array less than threads, loop will only execute once.
    for (int i = tid; i < arraySize; i += blockDim.x * gridDim.x) {
        int bin = input[i];
        bin = static_cast<int>((bin - min_value) / bin_size);
        if (bin >= 0 && bin < numBins) {
            // atomicAdd is used to safely update the global bins array from multiple threads
            atomicAdd(&bins[bin], 1);
        }
    }
}

__global__ void hist(
    int* bins,
    const int numBins,
    const int bin_size,
    const int min_value,
    const int max_value,
    const int *__restrict__ input,
    const int arraySize
) {
    extern __shared__ int shared_bins[];

    // Initialize shared memory bins to zero
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        shared_bins[i] = 0;
    }
    // Ensure all threads in the block have initialized shared memory before proceeding
    __syncthreads();

    // Each thread processes elements via grid-stride loop,
    // accumulating into block-local shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < arraySize; i += stride) {
        int bin = input[i];
        bin = static_cast<int>((bin - min_value) / bin_size);
        if (bin >= 0 && bin < numBins) {
            // atomicAdd is used to safely update the shared memory bins array from multiple threads
            // this atomic have affect only on block level and not on global level
            atomicAdd(&shared_bins[bin], 1);
        }
    }
    __syncthreads();

    // Merge block-local histogram into global bins
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&bins[i], shared_bins[i]);
    }
}

void launchHist(
    int* bins,
    const int numBins,
    const int bin_size,
    const int min_value,
    const int max_value,
    const int *__restrict__ input,
    const int input_size,
    int threadsPerBlock
) {
    int blocksPerGrid = cuda::ceil_div(input_size, threadsPerBlock);
    size_t sharedMemSize = numBins * sizeof(int);
    hist<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        bins, numBins, bin_size, min_value, max_value, input, input_size
    );
}

void launchNaiveHist(
    int* bins,
    const int numBins,
    const double bin_size,
    const int min_value,
    const int max_value,
    const int *__restrict__ input,
    const int arraySize,
    int threadsPerBlock
) {
    int blocksPerGrid = cuda::ceil_div(arraySize, threadsPerBlock);
    naiveHist<<<blocksPerGrid, threadsPerBlock>>>(
        bins, numBins, bin_size, min_value, max_value, input, arraySize
    );
}