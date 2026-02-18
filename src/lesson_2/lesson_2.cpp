#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "lesson_2.h"
#include "../../src_cuda/hist.cuh"

void histogram(bool is_naive = false);

void lesson_2() {
    std::cout << "Lesson 2" << std::endl;
    histogram();
    hist_benchmark();
}

void hist_benchmark() {
    std::cout << "\n=== Histogram Benchmark: Naive vs Optimized ===" << std::endl;

    int numBins = 256;
    int min_value = 0;
    int max_value = 256;
    int arraySize = 50'000'000; // 50 million elements
    int threadsPerBlock = 256;

    std::cout << "Array size: " << arraySize << " elements, Bins: " << numBins << std::endl;

    int* bins_naive = (int*)malloc(numBins * sizeof(int));
    int* bins_opt = (int*)malloc(numBins * sizeof(int));
    int* input = (int*)malloc(arraySize * sizeof(int));

    for (int i = 0; i < arraySize; i++) {
        input[i] = rand() % (max_value - min_value) + min_value;
    }

    int* d_bins;
    int* d_input;
    double bin_size = (double)(max_value - min_value) / numBins;

    cudaMalloc(&d_bins, numBins * sizeof(int));
    cudaMalloc(&d_input, arraySize * sizeof(int));
    cudaMemcpy(d_input, input, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // --- Naive Histogram Benchmark ---
    cudaEvent_t naive_start, naive_end;
    cudaEventCreate(&naive_start);
    cudaEventCreate(&naive_end);

    cudaMemset(d_bins, 0, numBins * sizeof(int));
    cudaEventRecord(naive_start);
    launchNaiveHist(d_bins, numBins, bin_size, min_value, max_value, d_input, arraySize, threadsPerBlock);
    cudaEventRecord(naive_end);
    cudaEventSynchronize(naive_end);

    float naive_ms = 0.0f;
    cudaEventElapsedTime(&naive_ms, naive_start, naive_end);
    cudaMemcpy(bins_naive, d_bins, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Optimized Histogram Benchmark ---
    cudaEvent_t opt_start, opt_end;
    cudaEventCreate(&opt_start);
    cudaEventCreate(&opt_end);

    cudaMemset(d_bins, 0, numBins * sizeof(int));
    cudaEventRecord(opt_start);
    launchHist(d_bins, numBins, bin_size, min_value, max_value, d_input, arraySize, threadsPerBlock);
    cudaEventRecord(opt_end);
    cudaEventSynchronize(opt_end);

    float opt_ms = 0.0f;
    cudaEventElapsedTime(&opt_ms, opt_start, opt_end);
    cudaMemcpy(bins_opt, d_bins, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    // --- CPU Histogram Benchmark ---
    int* bins_cpu = (int*)calloc(numBins, sizeof(int));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < arraySize; i++) {
        int bin = static_cast<int>((input[i] - min_value) / bin_size);
        if (bin >= numBins) bin = numBins - 1;
        bins_cpu[bin]++;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    // Verify results match
    bool match = true;
    for (int i = 0; i < numBins; i++) {
        if (bins_naive[i] != bins_opt[i]) {
            std::cerr << "GPU mismatch at bin " << i << ": naive=" << bins_naive[i] << " opt=" << bins_opt[i] << std::endl;
            match = false;
            break;
        }
        if (bins_cpu[i] != bins_opt[i]) {
            std::cerr << "CPU vs GPU mismatch at bin " << i << ": cpu=" << bins_cpu[i] << " gpu_opt=" << bins_opt[i] << std::endl;
            match = false;
            break;
        }
    }

    // Print results
    std::cout << "\nResults " << (match ? "MATCH" : "MISMATCH") << std::endl;
    std::cout << "CPU histogram time:       " << cpu_ms << " ms" << std::endl;
    std::cout << "Naive GPU histogram time: " << naive_ms << " ms" << std::endl;
    std::cout << "Optimized GPU hist time:  " << opt_ms << " ms" << std::endl;
    std::cout << "Speedup (CPU vs naive GPU):     " << cpu_ms / naive_ms << "x" << std::endl;
    std::cout << "Speedup (CPU vs optimized GPU): " << cpu_ms / opt_ms << "x" << std::endl;
    std::cout << "Speedup (naive GPU vs optimized GPU): " << naive_ms / opt_ms << "x" << std::endl;

    // Cleanup
    cudaEventDestroy(naive_start);
    cudaEventDestroy(naive_end);
    cudaEventDestroy(opt_start);
    cudaEventDestroy(opt_end);
    cudaFree(d_bins);
    cudaFree(d_input);
    free(bins_naive);
    free(bins_opt);
    free(bins_cpu);
    free(input);
}

void histogram(bool is_naive) {
    std::cout << "Histogram, is_naive: " << (is_naive ? "true" : "false") << std::endl;
    int numBins = 10;
    int min_value = 0;
    int max_value = 10;
    int arraySize = 10;
    int threadsPerBlock = 100;
    int* bins = (int*)malloc(numBins * sizeof(int));
    int* input = (int*)malloc(arraySize * sizeof(int));
    for (int i = 0; i < arraySize; i++) {
        input[i] = rand() % (max_value - min_value) + min_value;
    }

    for (int i = 0; i < arraySize; i++) {
        std::cout << " " << input[i] << " ";
    }
    std::cout << std::endl;

    int* d_bins;
    int* d_input;
    double bin_size = (max_value - min_value) / numBins;
    cudaMalloc(&d_bins, numBins * sizeof(int));
    cudaMalloc(&d_input, arraySize * sizeof(int));

    cudaMemcpy(d_input, input, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (is_naive) {
        launchNaiveHist(
            d_bins, numBins, bin_size, min_value, max_value, d_input, arraySize, threadsPerBlock
        );
    } else {
        launchHist(
            d_bins, numBins, bin_size, min_value, max_value, d_input, arraySize, threadsPerBlock
        );
    }
    cudaMemcpy(bins, d_bins, numBins * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numBins; i++) {
        std::cout << "value: " << i << ": " << bins[i] << std::endl;
    }
    cudaFree(d_bins);
    cudaFree(d_input);
    free(bins);
    free(input);
}

