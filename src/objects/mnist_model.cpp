#include "mnist_model.h"
#include "../utils/utils.h"


#include <cstdint>

void MNISTModel::backward(Tensor& input, std::vector<int>& targets, float lr) {
    // std::cout << "Backward" << std::endl;
    int batch_size = fc2.output->shape()[0];
    int num_classes = fc2.output->shape()[1];
    // std::cout << "Batch size: " << batch_size << std::endl;
    // print_vector("Output shape", fc2.output->shape());
    // print_vector("Targets", targets);

    auto grad = Tensor::create_zeros({ batch_size, num_classes });

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            float prob = fc2.output->get({i, j});
            float target = (targets[i] == j) ? 1.0f : 0.0f;
            float gradient = (prob - target) / batch_size;
            // std::cout << "Prob: " << prob << ", Target: " << target << ", Gradient: " << gradient << std::endl;
            grad->set({i, j}, gradient);
        }
        std::cout << std::endl;
    }
    // std::cout << "Grad: " << std::endl;
    // grad->print();
    grad->prepare_for_gpu_work();

    fc2.backward(*fc1.output, *grad, lr);

    // Backprop through ReLU
    // The gradient of ReLU is 1 if x > 0, else 0.
    // fc2.grad_input now contains the gradient w.r.t. fc1's output
    TensorOperation::relu_backward_inplace(*fc2.grad_input, *fc1.output);
    fc1.backward(input, *fc2.grad_input, lr);
}

float MNISTModel::sgd_step(Tensor& input, std::vector<int>& targets, float lr) {
    forward(input);
    backward(input, targets, lr);
    float loss = TensorOperation::cross_entropy_loss_cpu(*fc2.output, targets);
    return loss;
}

std::vector<int> MNISTModel::predict(Tensor& input) {
    forward(input);
    print_vector("Predictions", fc2.output->shape());
    fc2.output->print();
    std::vector<int> predictions;
    for (int i = 0; i < fc2.output->shape()[0]; i++) {
        int max_val_index = std::max_element(
            fc2.output->data() + i * fc2.output->shape()[1],
            fc2.output->data() + (i + 1) * fc2.output->shape()[1]
        ) - fc2.output->data();
        predictions.push_back(max_val_index - fc2.output->shape()[1] * i);
    }
    return predictions;
}