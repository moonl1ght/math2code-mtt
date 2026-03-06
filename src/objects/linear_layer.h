#pragma once

#include <iostream>
#include "tensor.h"

class LinearLayer {
public:
    std::unique_ptr<Tensor> weights; // Shape: (out_features, in_features)
    std::unique_ptr<Tensor> bias; // Shape: (1, out_features)

    LinearLayer(int in_features, int out_features) {
        weights = Tensor::create_random({out_features, in_features});

        bias = Tensor::create_random({1, out_features});
    }
};