#!/usr/bin/env python3
"""Downloads the MNIST dataset and saves raw binary files readable from C++.

Output files in data/mnist/:
  train_images.bin  -- uint32 n, uint32 rows, uint32 cols, then n*rows*cols uint8 pixels
  train_labels.bin  -- uint32 n, then n uint8 labels
  test_images.bin
  test_labels.bin

C++ reading example:
  FILE* f = fopen("data/mnist/train_images.bin", "rb");
  uint32_t n, rows, cols;
  fread(&n, 4, 1, f);
  fread(&rows, 4, 1, f);
  fread(&cols, 4, 1, f);
  std::vector<uint8_t> pixels(n * rows * cols);
  fread(pixels.data(), 1, pixels.size(), f);
  fclose(f);
"""

import os
import gzip
import struct
import urllib.request
import numpy as np

MIRROR_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
BASE_URL   = "http://yann.lecun.com/exdb/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnist")


def download_file(filename):
    dest = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(dest):
        print(f"  {filename} already exists, skipping.")
        return dest
    for base in [MIRROR_URL, BASE_URL]:
        url = base + filename
        try:
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:
            print(f"  Failed ({e}), trying next mirror...")
    raise RuntimeError(f"Could not download {filename}")


def parse_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x803
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def parse_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x801
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def save_images_bin(images, path):
    """uint32 n, uint32 rows, uint32 cols, then raw uint8 pixels (row-major)."""
    n, rows, cols = images.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<III", n, rows, cols))
        f.write(images.astype(np.uint8).tobytes())
    print(f"  Saved {path}  ({n} images, {rows}x{cols})")


def save_labels_bin(labels, path):
    """uint32 n, then raw uint8 labels."""
    n = len(labels)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(labels.astype(np.uint8).tobytes())
    print(f"  Saved {path}  ({n} labels)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    gz = {name: download_file(name) for name in FILES}

    print("\nParsing...")
    train_images = parse_images(gz["train-images-idx3-ubyte.gz"])
    train_labels = parse_labels(gz["train-labels-idx1-ubyte.gz"])
    test_images  = parse_images(gz["t10k-images-idx3-ubyte.gz"])
    test_labels  = parse_labels(gz["t10k-labels-idx1-ubyte.gz"])

    print("\nSaving binary files...")
    save_images_bin(train_images, os.path.join(OUTPUT_DIR, "train_images.bin"))
    save_labels_bin(train_labels, os.path.join(OUTPUT_DIR, "train_labels.bin"))
    save_images_bin(test_images,  os.path.join(OUTPUT_DIR, "test_images.bin"))
    save_labels_bin(test_labels,  os.path.join(OUTPUT_DIR, "test_labels.bin"))

    print("\nDone.")


if __name__ == "__main__":
    main()
