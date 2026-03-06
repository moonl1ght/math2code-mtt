#pragma once

#include "linear_layer.h"
#include "tensor.h"

class MNISTModel {
public:
    LinearLayer fc1; // 784 -> 128
    LinearLayer fc2; // 128 -> 10

    MNISTModel() : fc1(784, 128), fc2(128, 10) {}
};