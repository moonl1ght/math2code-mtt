#include "lesson_4.h"
#include <iostream>

#include "../utils/utils.h"
#include "../objects/mnist.h"

void lesson_4() {
    std::cout << "Lesson 4" << std::endl;

    auto train_data = MnistReader::load_train();
    std::cout << "Train data: " << train_data.images->shape()[0] << std::endl;
    std::cout << "Train data labels: " << train_data.labels.size() << std::endl;
    print_vector("Train data images", train_data.images->shape());
}