#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include "../../src_cuda/vector_add_kernel.cuh"
#include "lesson_1.h"

void vec_add();
void vec_add_benchmark();
void device_info();

void lesson_1()
{
    std::cout << "Lesson 1" << std::endl;

    vec_add();
    vec_add_benchmark();
    device_info();
}

void vec_add()
{
    std::cout << "Vector Addition using CUDA (CPU + GPU)" << std::endl;

    int numElements = 1000;
    size_t size = numElements * sizeof(float);

    // 1. Allocate Host (CPU) memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (!h_A || !h_B || !h_C)
    {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return;
    }

    // Initialize data
    std::cout << "Initializing data..." << std::endl;
    for (int i = 0; i < numElements; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 2. Allocate Device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. Copy data from Host to Device
    std::cout << "Copying data to GPU..." << std::endl;
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. Launch Kernel (via wrapper function)
    std::cout << "Running CUDA kernel..." << std::endl;
    int threadsPerBlock = 256;
    launchVectorAdd(d_A, d_B, d_C, numElements, threadsPerBlock);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for kernel errors
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 5. Copy result back to Host
    std::cout << "Copying results back to CPU..." << std::endl;
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 6. Verify results
    std::cout << "Verifying results..." << std::endl;
    bool success = true;
    for (int i = 0; i < numElements; i++)
    {
        if (h_C[i] != 3.0f)
        {
            std::cerr << "Error at index " << i << ": " << h_C[i] << " != 3.0" << std::endl;
            success = false;
            break;
        }
    }

    if (success)
    {
        std::cout << "Success! " << numElements << " vector additions completed correctly." << std::endl;
    }

    // 7. Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

void vec_add_benchmark()
{
    std::cout << "\n=== Vector Addition Benchmark: CPU vs GPU ===" << std::endl;

    int numElements = 50'000'000; // 50 million elements
    size_t size = numElements * sizeof(float);

    std::cout << "Vector size: " << numElements << " elements (" << size / (1024 * 1024) << " MB)" << std::endl;

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);

    // Initialize data
    for (int i = 0; i < numElements; i++)
    {
        h_A[i] = static_cast<float>(i) * 0.5f;
        h_B[i] = static_cast<float>(i) * 0.3f;
    }

    // --- CPU Benchmark ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numElements; i++)
    {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --- GPU Benchmark ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // GPU timing with CUDA events (includes memory transfers)
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaEventRecord(gpu_start);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    launchVectorAdd(d_A, d_B, d_C, numElements, threadsPerBlock);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end);

    // GPU timing kernel-only (no memory transfers)
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    cudaEventRecord(kernel_start);
    launchVectorAdd(d_A, d_B, d_C, numElements, threadsPerBlock);
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_end);

    // Verify GPU results match CPU
    bool match = true;
    for (int i = 0; i < numElements; i++)
    {
        if (std::abs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5f)
        {
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_C_cpu[i] << " GPU=" << h_C_gpu[i] << std::endl;
            match = false;
            break;
        }
    }

    // Print results
    std::cout << "\nResults " << (match ? "MATCH" : "MISMATCH") << std::endl;
    std::cout << "CPU time:                  " << cpu_ms << " ms" << std::endl;
    std::cout << "GPU time (with transfers): " << gpu_ms << " ms" << std::endl;
    std::cout << "GPU time (kernel only):    " << kernel_ms << " ms" << std::endl;
    std::cout << "Speedup (with transfers):  " << cpu_ms / gpu_ms << "x" << std::endl;
    std::cout << "Speedup (kernel only):     " << cpu_ms / kernel_ms << "x" << std::endl;

    // Cleanup
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
}

void device_info()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Number of SM (Streaming Multiprocessors): " << prop.multiProcessorCount << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Maximum number of threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum number of threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Maximum number of threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}