#include "TensorOperation.h"

#include "../../src_cuda/add.cuh"
#include "../../src_cuda/relu.cuh"
#include "../../src_cuda/sumrow.cuh"
#include "../../src_cuda/add_weighted.cuh"
#include "../../src_cuda/relu_backward.cuh"

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
        int batch_size = A.shape()[0];
        int num_classes = A.shape()[1];
        for (int i = 0; i < batch_size; i++) {
            float max_val = *std::max_element(A.data() + i * num_classes, A.data() + (i + 1) * num_classes);

            float sum = 0.0f;
            for (int j = 0; j < num_classes; j++) {
                float val = A.get({i, j});
                val = std::exp(val - max_val); // Subtract max_val for stability
                sum += val;
                A.set({i, j}, val);
            }

            for (int j = 0; j < num_classes; j++) {
                A.set({i, j}, A.get({i, j}) / sum);
            }
        }
    }

    void sum_rows_inplace_cpu(const Tensor& A, Tensor& B) {
        int batch_size = A.shape()[0];
        int num_classes = A.shape()[1];
        for (int i = 0; i < batch_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < num_classes; j++) {
                sum += A.get({i, j});
            }
            B.set({i}, sum);
        }
    }

    void sum_rows_inplace(const Tensor& A, Tensor& B) {
        if (!A.is_gpu_allocated() || !B.is_gpu_allocated()) {
            throw std::runtime_error("GPU memory not allocated");
        }
        if (A.shape().back() != B.shape().back()) {
            throw std::runtime_error("Incompatible shapes for summing rows");
        }
        launchSumRowsInplace(A.gpu_data(), B.gpu_data(), A.shape().back(), A.shape()[A.shape().size() - 2]);
    }

    void add_weighted_inplace(Tensor& A, const Tensor& B, float learning_rate) {
        if (!A.is_gpu_allocated() || !B.is_gpu_allocated()) {
            throw std::runtime_error("GPU memory not allocated");
        }
        if (A.shape() != B.shape()) {
            throw std::runtime_error("Incompatible shapes for adding weighted");
        }
        launchAddWeightedInplace(A.gpu_data(), B.gpu_data(), A.total_size(), learning_rate);
    }

    void relu_backward_inplace(Tensor& grad, const Tensor& output) {
        if (!grad.is_gpu_allocated() || !output.is_gpu_allocated()) {
            throw std::runtime_error("GPU memory not allocated");
        }
        if (grad.shape() != output.shape()) {
            throw std::runtime_error("Incompatible shapes for relu backward");
        }
        launchReluBackwardInplace(grad.gpu_data(), output.gpu_data(), grad.total_size());
    }

    float cross_entropy_loss_cpu(const Tensor& predictions, const std::vector<int>& labels) {
        int batch_size = predictions.shape()[0];
        int num_classes = predictions.shape()[1];
        float loss = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < num_classes; j++) {
                float prob = predictions.get({i, j});
                float target = (labels[i] == j) ? 1.0f : 0.0f;
                loss += -std::log(prob) * target;
            }
        }
        return loss / batch_size;
    }
}