#include "lesson_3.h"
#include "../../src/objects/tensor.h"

void tensor_test();

void lesson_3() {
    std::cout << "Lesson 3" << std::endl;
    tensor_test();
}

void tensor_test() {
    std::cout << "Tensor Test" << std::endl;
    auto tensor = Tensor::create_random({2, 2, 2});
    tensor->print(true);
    tensor->print();
}