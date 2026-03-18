#include "linear_layer.h"

void LinearLayer::backward(Tensor& input, Tensor& grad, float learning_rate) {
    // std::cout << "Backward " << name << std::endl;
    //grad shape : (batch_size, out_features)
    //input shape : (batch_size, in_features)

    // Calculate grad_weights = input ^ T * grad
    // print_vector("Input shape", input.shape());
    // print_vector("Grad shape", grad.shape());
    // print_vector("Weights shape", weights->shape());
    MatmulOperation::matmul_inplace(input, grad, *grad_weights, true, false);

    // std::cout << "Grad weights: " << std::endl;
    // grad_weights->print();
    // print_vector("Grad weights shape", grad_weights->shape());

    // Calculate grad_bias = sum(grad) over batch
    TensorOperation::sum_rows_inplace(grad, *grad_bias);
    // std::cout << "Grad bias: " << std::endl;
    // grad_bias->print();
    // print_vector("Grad bias shape", grad_bias->shape());

    // Calculate grad_input = grad * weights^T (for the next layer back)
    MatmulOperation::matmul_inplace(grad, *weights, *grad_input, false, true);
    // std::cout << "Grad input: " << std::endl;
    // grad_input->print();
    // print_vector("Grad input shape", grad_input->shape());

    // Update parameters (SGD step)
    TensorOperation::add_weighted_inplace(*weights, *grad_weights, learning_rate);
    TensorOperation::add_weighted_inplace(*bias, *grad_bias, learning_rate);
    // weights->copy_to_host();
    // bias->copy_to_host();

    // std::cout << "Weights: " << std::endl;
    // weights->print();
//     std::cout << "Bias: " << std::endl;
//     bias->print();
//     std::cout << "Grad weights: " << std::endl;
//     grad_weights->print();
//     std::cout << "Grad bias: " << std::endl;
}