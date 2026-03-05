#include "lesson_3.h"
#include "../../src/objects/tensor.h"
#include "../../src/operations/Operations.h"

void tensor_test();
void tensor_matmul_naive_test();
void tensor_matmul_test();

void lesson_3() {
    std::cout << "Lesson 3" << std::endl;
    // tensor_matmul_naive_test();
    tensor_matmul_test();
}

void tensor_test() {
    std::cout << "Tensor Test" << std::endl;
    auto tensor = Tensor::create_random({2, 2, 2});
    tensor->print(true);
    tensor->print();
}

void tensor_matmul_naive_test() {
    std::cout << "Tensor Matmul Naive Test" << std::endl;
    
    auto A = std::make_unique<Tensor>(std::vector<int>{2, 2}, std::vector<float>{1, 2, 3, 4});
    auto B = std::make_unique<Tensor>(std::vector<int>{2, 2}, std::vector<float>{5, 6, 7, 8});
    auto C = Operations::naive_matmul_nd(*A, *B);
    std::cout << "A: " << std::endl;
    A->print(true);
    A->print();
    std::cout << "B: " << std::endl;
    B->print(true);
    B->print();
    std::cout << "C: " << std::endl;
    C->print(true);
    C->print();
}

void tensor_matmul_test() {
    std::cout << "Tensor Matmul Test" << std::endl;
    
    auto A = std::make_unique<Tensor>(std::vector<int>{1,3, 5, 5});
    auto B = std::make_unique<Tensor>(std::vector<int>{3, 1,5, 5});
    auto C = Operations::matmul_nd(*A, *B);
    // std::cout << "A: " << std::endl;
    // A->print(true);
    // A->print();
    // std::cout << "B: " << std::endl;
    // B->print(true);
    // B->print();
    // std::cout << "C: " << std::endl;
    // C->print(true);
    // C->print();
}