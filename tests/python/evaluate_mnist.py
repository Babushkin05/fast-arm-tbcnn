#!/usr/bin/env python3
"""Evaluate MNIST model accuracy on real test data from Kaggle."""

import sys
import os
import gzip
import struct

script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', '..', 'tbn-runtime', 'build', 'python')
sys.path.insert(0, build_dir)

import tbn
import numpy as np

# MNIST normalization
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def load_mnist_from_kaggle():
    """Load MNIST from Kaggle dataset."""
    import kagglehub

    print("Downloading MNIST from Kaggle...")
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print(f"Dataset path: {path}")

    # Read test images
    images_file = os.path.join(path, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
    labels_file = os.path.join(path, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

    # Try alternative paths
    if not os.path.exists(images_file):
        images_file = os.path.join(path, 't10k-images.idx3-ubyte')
    if not os.path.exists(images_file):
        # Find any file with "images" in name
        for f in os.listdir(path):
            if 'images' in f.lower() and 'idx3' in f.lower():
                images_file = os.path.join(path, f)
                break

    if not os.path.exists(labels_file):
        labels_file = os.path.join(path, 't10k-labels.idx1-ubyte')
    if not os.path.exists(labels_file):
        for f in os.listdir(path):
            if 'labels' in f.lower() and 'idx1' in f.lower():
                labels_file = os.path.join(path, f)
                break

    print(f"Images file: {images_file}")
    print(f"Labels file: {labels_file}")

    # Check if gzipped
    open_func = open
    if images_file.endswith('.gz'):
        open_func = gzip.open
        images_file = images_file[:-3] if not os.path.exists(images_file) else images_file
    if labels_file.endswith('.gz'):
        labels_file = labels_file[:-3] if not os.path.exists(labels_file) else labels_file

    # Actually try to find the files
    print(f"Files in dataset: {os.listdir(path)}")

    # Read with correct opener
    with open(images_file, 'rb') if os.path.exists(images_file) else gzip.open(images_file + '.gz', 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with open(labels_file, 'rb') if os.path.exists(labels_file) else gzip.open(labels_file + '.gz', 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels

def main():
    # Load MNIST test data
    X_test, y_test = load_mnist_from_kaggle()
    print(f"Test set size: {len(X_test)}\n")

    # Load model
    model_path = os.path.join(script_dir, '..', 'models', 'mnist', 'mnist-8.onnx')
    print(f"Loading model: {model_path}")
    model = tbn.load_model(model_path)
    print("Model loaded!\n")

    # Test on subset (for speed)
    test_size = 1000
    print(f"Testing on {test_size} samples...")

    correct = 0
    for i in range(test_size):
        # Get image
        img = X_test[i].astype(np.float32) / 255.0
        img = (img - MNIST_MEAN) / MNIST_STD
        img = img.reshape(1, 1, 28, 28)

        # Predict
        output = model.run(img)
        pred = np.argmax(output[0])

        if pred == y_test[i]:
            correct += 1

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{test_size} - Current accuracy: {correct/(i+1):.2%}")

    accuracy = correct / test_size
    print(f"\n{'='*50}")
    print(f"ACCURACY: {accuracy:.2%} ({correct}/{test_size})")
    print(f"{'='*50}")

    return accuracy

if __name__ == '__main__':
    main()
