#include "lesson_4.h"
#include <iostream>

#include "../utils/utils.h"
#include "../objects/mnist.h"
#include "../objects/tensor.h"
#include "../operations/TensorOperation.h"
#include "../objects/mnist_model.h"

void add_test();
void relu_test();
void sum_rows_test();
void mnist_model_test();
void mnist_test_train();
void mnist_train();

void lesson_4() {
    std::cout << "Lesson 4" << std::endl;

    mnist_train();
}

void mnist_train() {
    std::cout << "MNIST Train" << std::endl;
    int batch_size = 60;
    auto train_data = MnistReader::load_train(batch_size);
    train_data.prepare_for_gpu_work();
    std::cout << "Train data: " << train_data.images.size() << std::endl;
    std::cout << "Train data labels: " << train_data.labels.size() << std::endl;
    print_vector("Train data images", train_data.images[0]->shape());
    MNISTModel model(batch_size);
    model.prepare_for_gpu_work();

    int epochs = 10;
    int steps = train_data.images.size();
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch << std::endl;
        for (size_t i = 0; i < steps; i++) {
            auto& input = train_data.images[i];
            std::vector<int> targets;
            for (size_t j = 0; j < input->shape()[0]; j++) {
                targets.push_back(train_data.labels[i * batch_size + j]);
            }
            // print_vector("Targets", targets);
            float loss = model.sgd_step(*input, targets, 0.001f);
            std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
        }
    }

    auto test_data = MnistReader::load_test(batch_size);
    test_data.prepare_for_gpu_work();
    std::cout << "Test data: " << test_data.images.size() << std::endl;
    std::cout << "Test data labels: " << test_data.labels.size() << std::endl;
    print_vector("Test data images", test_data.images[0]->shape());
    auto predictions = model.predict(*test_data.images[0]);
    print_vector("Predictions", predictions);
    std::cout << "Label: " << test_data.labels[0] << std::endl;
}

void mnist_test_train() {
    std::cout << "MNIST Test Train" << std::endl;

    MNISTModel model(8, 4, 2, 2);
    model.prepare_for_gpu_work();

    std::vector<int> targets = {0, 1};
    auto input = Tensor::create_random({2, 8});
    std::cout << "Input: " << std::endl;
    input->print();
    input->prepare_for_gpu_work();
    float loss = model.sgd_step(*input, targets, 0.01f);
    std::cout << "Loss: " << loss << std::endl;
}

void mnist_model_test() {
    std::cout << "MNIST Model Test" << std::endl;
    auto train_data = MnistReader::load_train(10);
    train_data.prepare_for_gpu_work();
    std::cout << "Train data: " << train_data.images.size() << std::endl;
    std::cout << "Train data labels: " << train_data.labels.size() << std::endl;
    print_vector("Train data images", train_data.images[0]->shape());

    MNISTModel model(10);
    model.prepare_for_gpu_work();
    train_data.images[0]->prepare_for_gpu_work();
    auto predictions = model.predict(*train_data.images[0]);
    print_vector("Predictions", predictions);
}

void add_test() {
    std::cout << "Add Test" << std::endl;
    auto A = std::make_unique<Tensor>(std::vector<int>{2, 2}, std::vector<float>{1, 2, 5, 4});
    A->prepare_for_gpu_work();
    auto B = std::make_unique<Tensor>(std::vector<int>{1, 2}, std::vector<float>{5, 6});
    B->prepare_for_gpu_work();
    TensorOperation::add_inplace(*A, *B);
    A->copy_to_host();
    B->copy_to_host();
    A->print();
    B->print();
}

void relu_test() {
    std::cout << "Relu Test" << std::endl;
    auto A = std::make_unique<Tensor>(std::vector<int>{2, 2}, std::vector<float>{-1, 2, 3, -4});
    A->prepare_for_gpu_work();
    TensorOperation::relu_inplace(*A);
    A->copy_to_host();
    A->print();
}

void sum_rows_test() {
    std::cout << "Sum Rows Test" << std::endl;
    auto A = std::make_unique<Tensor>(std::vector<int>{2, 6}, std::vector<float>{7, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    A->prepare_for_gpu_work();
    auto B = std::make_unique<Tensor>(std::vector<int>{1, 6}, std::vector<float>{0, 0, 0, 0, 0, 0});
    B->prepare_for_gpu_work();
    TensorOperation::sum_rows_inplace(*A, *B);
    B->copy_to_host();
    B->print();
}