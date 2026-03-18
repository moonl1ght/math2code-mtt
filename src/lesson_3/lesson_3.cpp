#include "lesson_3.h"
#include "../../src/objects/tensor.h"
#include "../../src/operations/MatmulOperation.h"

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
    auto tensor = Tensor::create_random({ 2, 2, 2 });
    tensor->print(true);
    tensor->print();
}

void tensor_matmul_naive_test() {
    std::cout << "Tensor Matmul Naive Test" << std::endl;

    auto A = std::make_unique<Tensor>(std::vector<int>{2, 2}, std::vector<float>{1, 2, 3, 4});
    auto B = std::make_unique<Tensor>(std::vector<int>{2, 2}, std::vector<float>{5, 6, 7, 8});
    auto C = MatmulOperation::naive_matmul_nd(*A, *B);
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

    auto A = Tensor::arrange_contiguous({ 1, 3, 256, 256 });
    A->prepare_for_gpu_work();
    auto B = Tensor::arrange_contiguous({ 3, 1 ,256, 256 });
    B->prepare_for_gpu_work();
    // auto C_inplace = Tensor::create_zeros({ 4, 2 });
    // C_inplace->prepare_for_gpu_work();
    // std::cout << "A: " << std::endl;
    // A->print(true);
    // A->print();
    // std::cout << "B: " << std::endl;
    // B->print(true);
    // B->print();
    auto C_cpu = MatmulOperation::matmul_cpu(*A, *B, true, false);
    auto C = MatmulOperation::matmul_nd(*A, *B, true, false);
    // MatmulOperation::matmul_inplace(*A, *B, *C_inplace, true, false);

    C->copy_to_host();
    // C_inplace->copy_to_host();
    // std::cout << "C: " << std::endl;
    // C->print(true);
    // C->print();
    // std::cout << "C_inplace: " << std::endl;
    // C_inplace->print(true);
    // C_inplace->print();
    // std::cout << "C_cpu: " << std::endl;
    // C_cpu->print(true);
    // C_cpu->print();
    if (*C == *C_cpu) {
        std::cout << "C == C_cpu" << std::endl;
    }
    else {
        std::cout << "C != C_cpu" << std::endl;
    }
    // std::cout << "A: " << std::endl;
    // A->print(true);
    // A->print();
    // std::cout << "B: " << std::endl;
    // B->print(true);
    // B->print();
    // std::cout << "C: " << std::endl;
    // C->print(true);
    // C->print();
    // std::cout << "C_cpu: " << std::endl;
    // C_cpu->print(true);
    // C_cpu->print();
}