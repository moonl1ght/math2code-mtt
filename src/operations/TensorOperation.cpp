#include "TensorOperation.h"

#include "../../src_cuda/add.cuh"
#include "../../src_cuda/relu.cuh"

#include <algorithm>

#define MAX_THREADS_PER_BLOCK 256

namespace TensorOperation {
    bool check_broadcast_compatibility(const std::vector<int>& shapeA, const std::vector<int>& shapeB) {
        auto itA = shapeA.rbegin();
        auto itB = shapeB.rbegin();

        while (itA != shapeA.rend() && itB != shapeB.rend()) {
            int dimA = *itA;
            int dimB = *itB;

            // Rule: Dimensions must match, or one must be 1
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                std::cerr << "Incompatible dimensions: " << dimA << " and " << dimB << std::endl;
                return false;
            }

            itA++;
            itB++;
        }
        return true;
    }

    void add_inplace(Tensor& A, const Tensor& B) {
        if (!A.is_gpu_allocated() || !B.is_gpu_allocated()) {
            throw std::runtime_error("GPU memory not allocated");
        }
        if (!check_broadcast_compatibility(A.shape(), B.shape())) {
            throw std::runtime_error("Incompatible shapes for broadcasting");
        }

        launchAddInplace(A.gpu_data(), B.gpu_data(), A.total_size(), B.total_size(), MAX_THREADS_PER_BLOCK);
    }

    void relu_inplace(Tensor& A) {
        if (!A.is_gpu_allocated()) {
            throw std::runtime_error("GPU memory not allocated");
        }
        launchReluInplace(A.gpu_data(), A.total_size(), MAX_THREADS_PER_BLOCK);
    }

    void softmax_inplace_cpu(Tensor& A) {
        float max_val = *std::max_element(A.data(), A.data() + A.total_size());

        float sum = 0.0f;
        for (int i = 0; i < A.total_size(); i++) {
            float val = A.data()[i];
            val = std::exp(val - max_val); // Subtract max_val for stability
            sum += val;
            A.data()[i] = val;
        }

        for (int i = 0; i < A.total_size(); i++) {
            A.data()[i] = A.data()[i] / sum;
        }
    }
}