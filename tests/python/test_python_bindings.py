#!/usr/bin/env python3
"""Test TBN Python bindings with MNIST model."""

import sys
import os

# Add build directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', 'tbn-runtime', 'build', 'python')
sys.path.insert(0, build_dir)

import tbn
import numpy as np

def main():
    print('=== TBN Python Bindings Test ===\n')

    # Load MNIST model
    model_path = os.path.join(script_dir, '..', 'tests', 'models', 'mnist', 'mnist-8.onnx')
    print(f'Loading model: {model_path}')
    model = tbn.load_model(model_path)

    print(f'Input names: {model.input_names()}')
    print(f'Output names: {model.output_names()}')
    print(f'Input shape: {model.input_shape("Input3")}')
    print()

    # Create dummy input
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    print(f'Input shape: {dummy_input.shape}')

    # Run inference
    print('Running inference...')
    output = model.run(dummy_input)

    print(f'Output shape: {output.shape}')
    print(f'Output values: {output[0]}')
    print(f'Predicted digit: {np.argmax(output)}')
    print()

    # Softmax for confidence
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    probs = softmax(output[0])
    predicted = np.argmax(probs)
    confidence = probs[predicted]

    print(f'=== Result ===')
    print(f'Predicted digit: {predicted}')
    print(f'Confidence: {confidence:.2%}')
    print()
    print('All probabilities:')
    for i, p in enumerate(probs):
        bar = '█' * int(p * 30)
        print(f'  {i}: {p:.2%} {bar}')

if __name__ == '__main__':
    main()
