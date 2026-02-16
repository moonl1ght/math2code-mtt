# math2code-mtt

MNIST to Transformer series: Building Neural Networks from scratch with C++ and CUDA.

An educational project that implements neural network components from the ground up using C++ for CPU code and CUDA for GPU acceleration - progressing from GPU fundamentals all the way to transformers.

## Lessons

### Lesson 1 - GPU Fundamentals: Vector Addition

Introduces core CUDA concepts through vector addition:

- **Vector Add** - Allocates host/device memory, copies data to GPU, launches a CUDA kernel, copies results back, and verifies correctness (1,000 elements).
- **Benchmark** - Compares CPU vs GPU performance on 50M elements, measuring total time (with memory transfers) and kernel-only time.
- **Device Info** - Queries and displays GPU properties (SMs, compute capability, memory, threading limits).

## Prerequisites

- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed at `/usr/local/cuda`
- g++ with C++20 support
- GNU Make

## Building and Running

```bash
make          # Build the project
make run      # Build and run
make clean    # Remove build files
make rebuild  # Clean and rebuild
make info     # Show compiler info
```

The default CUDA architecture is `sm_89` (RTX 4000 series / Ada). Change `CUDA_ARCH` in the [Makefile](Makefile) to match your GPU - use `nvcc --list-gpu-code` to see supported values.

## License

[MIT](LICENSE)
