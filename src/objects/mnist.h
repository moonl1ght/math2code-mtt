#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.h"

class MnistReader {
public:
    struct Dataset {
        // images: shape [n, rows, cols], pixel values normalized to [0, 1]
        std::vector<std::unique_ptr<Tensor>> images;
        // labels: shape [n], integer class 0-9
        std::vector<uint8_t> labels;

        void prepare_for_gpu_work() {
            for (auto& image : images) {
                image->prepare_for_gpu_work();
            }
        }
    };

    // Load both images and labels from the binary files produced by download_mnist.py
    static Dataset load(const std::string& images_path, const std::string& labels_path, int batch_size = 100) {
        Dataset ds;
        ds.images = read_images(images_path, batch_size);
        ds.labels = read_labels(labels_path);

        if (static_cast<int>(ds.labels.size()) != ds.images.size() * batch_size) {
            throw std::runtime_error("Image and label counts do not match");
        }
        return ds;
    }

    // Convenience: load train split relative to a data root dir
    static Dataset load_train(int batch_size = 100, const std::string& data_dir = "data/mnist") {
        return load(data_dir + "/train_images.bin", data_dir + "/train_labels.bin", batch_size);
    }

    // Convenience: load test split relative to a data root dir
    static Dataset load_test(int batch_size = 100, const std::string& data_dir = "data/mnist") {
        return load(data_dir + "/test_images.bin", data_dir + "/test_labels.bin", batch_size);
    }

private:
    // Binary layout: [uint32 n][uint32 rows][uint32 cols][n*rows*cols uint8 pixels]
    static std::vector<std::unique_ptr<Tensor>> read_images(const std::string& path, int batch_size = 100) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: " + path);

        uint32_t n, rows, cols;
        f.read(reinterpret_cast<char*>(&n),    4);
        f.read(reinterpret_cast<char*>(&rows), 4);
        f.read(reinterpret_cast<char*>(&cols), 4);

        std::vector<uint8_t> raw(n * rows * cols);
        f.read(reinterpret_cast<char*>(raw.data()), raw.size());
        if (!f) throw std::runtime_error("Unexpected EOF reading: " + path);

        std::vector<std::unique_ptr<Tensor>> images;

        // Normalize to [0, 1] and store in a Tensor shaped [n, rows, cols]
        for (size_t i = 0; i < n / batch_size; i++) {
            auto tensor = std::make_unique<Tensor>(std::vector<int>{
                batch_size,
                static_cast<int>(rows * cols)
            });
            for (size_t j = 0; j < rows * cols * batch_size; j++) {
                tensor->data()[j] = static_cast<float>(raw[i * batch_size * rows * cols + j]) / 255.0f;
            }
            images.push_back(std::move(tensor));
        }
        return images;
    }

    // Binary layout: [uint32 n][n uint8 labels]
    static std::vector<uint8_t> read_labels(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: " + path);

        uint32_t n;
        f.read(reinterpret_cast<char*>(&n), 4);

        std::vector<uint8_t> labels(n);
        f.read(reinterpret_cast<char*>(labels.data()), n);
        if (!f) throw std::runtime_error("Unexpected EOF reading: " + path);
        return labels;
    }
};
