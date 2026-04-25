#!/usr/bin/env python3
"""Download sample MNIST test images."""

import os
import urllib.request
import gzip
import struct
import numpy as np

def download_mnist_samples(output_dir='mnist_samples', num_samples=10):
    """Download MNIST test data and save sample images."""
    os.makedirs(output_dir, exist_ok=True)

    # Alternative URLs (original Yann LeCun site is often down)
    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    ]
    label_urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
        'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
    ]

    # Download test images
    images_file = os.path.join(output_dir, 't10k-images-idx3-ubyte.gz')
    labels_file = os.path.join(output_dir, 't10k-labels-idx1-ubyte.gz')

    print("Downloading MNIST test data...")

    if not os.path.exists(images_file):
        for url in urls:
            try:
                print(f"Trying: {url}")
                urllib.request.urlretrieve(url, images_file)
                break
            except Exception as e:
                print(f"Failed: {e}")
                continue

    if not os.path.exists(labels_file):
        for url in label_urls:
            try:
                print(f"Trying: {url}")
                urllib.request.urlretrieve(url, labels_file)
                break
            except Exception as e:
                print(f"Failed: {e}")
                continue

    if not os.path.exists(images_file) or not os.path.exists(labels_file):
        print("Failed to download MNIST data. Using fallback...")
        generate_fallback_samples(output_dir, num_samples)
        return

    print("Extracting...")

    # Read images
    with gzip.open(images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    # Read labels
    with gzip.open(labels_file, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Save samples for each digit
    print(f"Saving {num_samples} samples per digit...")

    for digit in range(10):
        digit_dir = os.path.join(output_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)

        indices = np.where(labels == digit)[0][:num_samples]
        for i, idx in enumerate(indices):
            img = images[idx]
            path = os.path.join(digit_dir, f'{digit}_{i:02d}.pgm')

            # Write PGM
            with open(path, 'wb') as f:
                f.write(f'P5\n{cols} {rows}\n255\n'.encode())
                f.write(img.tobytes())

    print(f"Done! Samples saved to {output_dir}/")
    print(f"Structure: {output_dir}/<digit>/<digit>_<num>.pgm")


def generate_fallback_samples(output_dir, num_samples=5):
    """Generate synthetic digit samples as fallback."""
    print(f"Generating {num_samples} synthetic samples per digit...")

    for digit in range(10):
        digit_dir = os.path.join(output_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)

        for sample in range(num_samples):
            # Create a simple representation of the digit
            img = np.zeros((28, 28), dtype=np.uint8)

            # Simple digit patterns
            patterns = {
                0: [(7, 5, 7, 22), (7, 5, 20, 5), (7, 22, 20, 22), (20, 5, 20, 22)],  # Oval
                1: [(5, 14, 22, 14)],  # Vertical line
                2: [(5, 7, 21), (5, 21, 10, 21), (10, 21, 20, 7), (10, 7, 20, 7), (14, 7, 20, 7)],  # 2 shape
                3: [(5, 7, 18), (5, 18, 14), (14, 14, 18), (14, 18, 22), (14, 7, 22)],  # 3 shape
                4: [(5, 7, 14, 7), (14, 5, 14, 22), (5, 14, 20, 14)],  # 4 shape
                5: [(5, 7, 18, 7), (5, 7, 5, 14), (5, 14, 15, 14), (15, 14, 15, 21), (15, 21, 5, 21)],  # 5 shape
                6: [(18, 7, 5, 7), (5, 7, 5, 21), (5, 21, 18, 21), (18, 21, 18, 14), (18, 14, 10, 14)],  # 6 shape
                7: [(5, 7, 22, 7), (22, 7, 10, 22)],  # 7 shape
                8: [(10, 7, 18, 7), (18, 7, 18, 12), (18, 12, 10, 12), (10, 12, 10, 7),  # Top circle
                    (10, 14, 18, 14), (18, 14, 18, 21), (18, 21, 10, 21), (10, 21, 10, 14)],  # Bottom circle
                9: [(10, 14, 18, 14), (18, 14, 18, 7), (18, 7, 10, 7), (10, 7, 10, 14), (18, 14, 18, 21)],  # 9 shape
            }

            # Draw pattern (simplified)
            for i in range(28):
                for j in range(28):
                    # Add some noise based on digit pattern
                    if digit == 0:
                        if 8 < i < 20 and 8 < j < 20 and not (12 < i < 16 and 12 < j < 16):
                            img[i, j] = 200 + np.random.randint(55)
                    elif digit == 1:
                        if 5 < i < 23 and 12 < j < 16:
                            img[i, j] = 200 + np.random.randint(55)
                    elif digit == 7:
                        if 5 < i < 9 and 5 < j < 23:
                            img[i, j] = 200 + np.random.randint(55)
                        if 8 < i < 20 and i - 5 < j < i:
                            img[i, j] = 200 + np.random.randint(55)
                    else:
                        # Generic digit pattern
                        center = 14
                        if abs(i - center) < 8 and abs(j - center) < 8:
                            img[i, j] = 150 + np.random.randint(100)

            # Add noise
            noise = np.random.randint(0, 30, (28, 28), dtype=np.uint8)
            img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

            path = os.path.join(digit_dir, f'{digit}_{sample:02d}.pgm')
            with open(path, 'wb') as f:
                f.write(f'P5\n28 28\n255\n'.encode())
                f.write(img.tobytes())

    print(f"Fallback samples saved to {output_dir}/")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'mnist_samples')
    download_mnist_samples(output_dir)
