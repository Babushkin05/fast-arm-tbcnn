#!/usr/bin/env python3
"""
MNIST Classifier using TBN Runtime

Usage:
    python mnist_classifier.py <image_path>
    python mnist_classifier.py --demo  # Run with sample images
"""

import sys
import os
import struct

# Add build directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', '..', 'tbn-runtime', 'build', 'python')
sys.path.insert(0, build_dir)

import tbn
import numpy as np

# MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_pgm(path):
    """Load PGM (Portable Gray Map) image format."""
    with open(path, 'rb') as f:
        # Read header
        magic = f.readline().decode('ascii').strip()
        if magic not in ('P5', 'P2'):
            raise ValueError(f"Not a PGM file (magic: {magic})")

        # Skip comments
        line = f.readline().decode('ascii').strip()
        while line.startswith('#'):
            line = f.readline().decode('ascii').strip()

        # Read dimensions
        width, height = map(int, line.split())

        # Read max value
        maxval = int(f.readline().decode('ascii').strip())

        if magic == 'P5':  # Binary format
            if maxval < 256:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            else:
                data = np.frombuffer(f.read(), dtype='>u2')
        else:  # P2 - ASCII format
            data = np.array([int(x) for x in f.read().decode('ascii').split()], dtype=np.uint8)

        return data.reshape((height, width))


def load_image(path):
    """Load image and convert to MNIST format (28x28 grayscale)."""
    try:
        # Try PGM format first (no dependencies)
        img = load_pgm(path)
    except (ValueError, UnicodeDecodeError):
        # Try with PIL
        from PIL import Image
        img = np.array(Image.open(path).convert('L'))

    # Resize to 28x28 if needed
    if img.shape != (28, 28):
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((28, 28), Image.Resampling.LANCZOS))

    return img.astype(np.float32)


def preprocess(img):
    """Preprocess image for MNIST model."""
    # Normalize to [0, 1]
    img = img / 255.0

    # Apply MNIST normalization
    img = (img - MNIST_MEAN) / MNIST_STD

    # Add batch dimension
    return img.reshape(1, 1, 28, 28).astype(np.float32)


def softmax(x):
    """Compute softmax probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def classify(model, image_path):
    """Classify a single image."""
    print(f"Loading: {image_path}")

    # Load and preprocess
    img = load_image(image_path)
    input_tensor = preprocess(img)

    # Run inference
    output = model.run(input_tensor)

    # Get prediction
    probs = softmax(output[0])
    predicted = np.argmax(probs)
    confidence = probs[predicted]

    return predicted, confidence, probs


def main():
    # Find model
    model_path = os.path.join(script_dir, '..', 'models', 'mnist', 'mnist-8.onnx')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Load model
    print(f"Loading model: {model_path}")
    model = tbn.load_model(model_path)
    print(f"Model loaded successfully!\n")

    # Check arguments
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print(f"  python {sys.argv[0]} digit.pgm")
        sys.exit(0)

    if sys.argv[1] == '--demo':
        # Demo mode - generate test images
        print("Demo mode: generating test digits...")

        # Create simple test images
        demo_dir = os.path.join(script_dir, 'demo_images')
        os.makedirs(demo_dir, exist_ok=True)

        # Generate a simple "7" pattern
        img = np.zeros((28, 28), dtype=np.uint8)
        img[5:8, 5:23] = 255  # Top horizontal
        img[7:12, 15:18] = 255  # Diagonal
        img[10:25, 12:15] = 255  # Vertical part

        demo_path = os.path.join(demo_dir, 'demo_7.pgm')
        # Write PGM
        with open(demo_path, 'wb') as f:
            f.write(f'P5\n28 28\n255\n'.encode())
            f.write(img.tobytes())

        print(f"Created demo image: {demo_path}\n")
        sys.argv = [sys.argv[0], demo_path]

    # Classify image
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    predicted, confidence, probs = classify(model, image_path)

    # Print result
    print(f"\n{'='*40}")
    print(f"PREDICTION: {predicted}")
    print(f"Confidence: {confidence:.2%}")
    print(f"{'='*40}\n")

    print("All probabilities:")
    for i, p in enumerate(probs):
        bar = '█' * int(p * 30)
        marker = ' <--' if i == predicted else ''
        print(f"  {i}: {p:.2%} {bar}{marker}")


if __name__ == '__main__':
    main()
