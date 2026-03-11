#include "lesson_4.h"
#include <iostream>

#include "../utils/utils.h"
#include "../objects/mnist.h"
#include "../objects/tensor.h"
#include "../operations/TensorOperation.h"
#include "../objects/mnist_model.h"

void add_test();
void relu_test();
void mnist_model_test();

void lesson_4() {
    std::cout << "Lesson 4" << std::endl;

    mnist_model_test();
}

void mnist_model_test() {
    std::cout << "MNIST Model Test" << std::endl;
    auto train_data = MnistReader::load_train(10);
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