#include "mnist_model.h"
#include "../utils/utils.h"

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