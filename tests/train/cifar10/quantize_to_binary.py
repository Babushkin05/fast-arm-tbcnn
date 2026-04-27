#!/usr/bin/env python3
"""
Post-training quantization: Convert float weights to binary.

This creates a model with binary weights that can be loaded directly
by TBN runtime without on-the-fly quantization overhead.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import os
from pathlib import Path


def quantize_weights_to_binary(weights: np.ndarray) -> np.ndarray:
    """Quantize float weights to binary {-1, +1}."""
    # Simple sign-based quantization
    binary = np.sign(weights)
    # Replace zeros with +1 (or could use -1)
    binary[binary == 0] = 1
    return binary.astype(np.float32)


def analyze_weight_distribution(weights: np.ndarray, name: str):
    """Print statistics about weight distribution."""
    zeros = np.sum(np.abs(weights) < 0.01)
    total = weights.size
    print(f"  {name}: mean={weights.mean():.4f}, std={weights.std():.4f}, "
          f"near_zero={zeros}/{total} ({100*zeros/total:.1f}%)")


def quantize_onnx_model(input_path: str, output_path: str):
    """
    Quantize all Conv and Linear weights in an ONNX model to binary.
    """
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    # Create a map of initializer names to their values
    initializers = {init.name: init for init in model.graph.initializer}

    quantized_count = 0

    # Find all Conv and MatMul nodes
    for node in model.graph.node:
        if node.op_type == 'Conv':
            # Conv weights are typically the second input
            if len(node.input) >= 2:
                weight_name = node.input[1]
                if weight_name in initializers:
                    init = initializers[weight_name]
                    weights = numpy_helper.to_array(init)

                    print(f"\nQuantizing Conv weight: {weight_name}")
                    print(f"  Shape: {weights.shape}")
                    analyze_weight_distribution(weights, weight_name)

                    # Quantize to binary
                    binary_weights = quantize_weights_to_binary(weights)

                    # Create new initializer with binary weights
                    new_init = TensorProto()
                    new_init.name = weight_name
                    new_init.dims.extend(weights.shape)
                    new_init.data_type = TensorProto.FLOAT
                    new_init.float_data.extend(binary_weights.flatten().tolist())

                    # Replace in graph
                    for i, init in enumerate(model.graph.initializer):
                        if init.name == weight_name:
                            model.graph.initializer[i].CopyFrom(new_init)
                            break

                    quantized_count += 1

        elif node.op_type == 'MatMul' or node.op_type == 'Gemm':
            # MatMul/Gemm weights are typically the second input (B matrix)
            if len(node.input) >= 2:
                weight_name = node.input[1]
                if weight_name in initializers:
                    init = initializers[weight_name]
                    weights = numpy_helper.to_array(init)

                    print(f"\nQuantizing MatMul/Gemm weight: {weight_name}")
                    print(f"  Shape: {weights.shape}")
                    analyze_weight_distribution(weights, weight_name)

                    # Quantize to binary
                    binary_weights = quantize_weights_to_binary(weights)

                    # Create new initializer
                    new_init = TensorProto()
                    new_init.name = weight_name
                    new_init.dims.extend(weights.shape)
                    new_init.data_type = TensorProto.FLOAT
                    new_init.float_data.extend(binary_weights.flatten().tolist())

                    # Replace in graph
                    for i, init in enumerate(model.graph.initializer):
                        if init.name == weight_name:
                            model.graph.initializer[i].CopyFrom(new_init)
                            break

                    quantized_count += 1

    # Validate model
    onnx.checker.check_model(model)

    # Save
    onnx.save(model, output_path)
    print(f"\n{'='*60}")
    print(f"Quantized {quantized_count} weight tensors")
    print(f"Saved binary-weight model to: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_path


def verify_accuracy(model_path: str, test_images: np.ndarray, test_labels: np.ndarray, num_samples: int = 100):
    """Verify that quantized model still works."""
    import onnxruntime as ort

    print(f"\nVerifying accuracy on {num_samples} samples...")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    sess = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    correct = 0
    for i in range(min(num_samples, len(test_images))):
        img = np.expand_dims(test_images[i], axis=0)
        output = sess.run(None, {input_name: img})[0]
        if np.argmax(output) == test_labels[i]:
            correct += 1

    accuracy = 100.0 * correct / min(num_samples, len(test_images))
    print(f"Accuracy: {accuracy:.1f}%")
    return accuracy


def load_test_data(num_samples: int = 100):
    """Load test data for verification."""
    import kagglehub
    from PIL import Image

    print("Loading test data...")
    path = kagglehub.dataset_download("ayush1220/cifar10")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)

    images = []
    labels = []

    test_path = os.path.join(path, 'cifar10', 'test')

    for class_name in class_names:
        class_path = os.path.join(test_path, class_name)
        class_idx = class_to_idx[class_name]

        img_files = sorted([f for f in os.listdir(class_path) if f.endswith('.png')])

        for img_name in img_files:
            if len(images) >= num_samples:
                break

            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = (img_array - mean) / std
            images.append(img_array)
            labels.append(class_idx)

        if len(images) >= num_samples:
            break

    return np.array(images[:num_samples], dtype=np.float32), np.array(labels[:num_samples])


def main():
    script_dir = Path(__file__).parent
    input_model = script_dir / 'cifar10_tbn.onnx'
    output_model = script_dir / 'cifar10_binary_weights.onnx'

    if not input_model.exists():
        print(f"Error: Input model not found: {input_model}")
        print("Please run train_tbn_cifar10.py first")
        return

    # Quantize weights
    quantize_onnx_model(str(input_model), str(output_model))

    # Verify accuracy
    try:
        test_images, test_labels = load_test_data(100)
        print(f"\nOriginal model accuracy:")
        verify_accuracy(str(input_model), test_images, test_labels)
        print(f"\nBinary-weight model accuracy:")
        verify_accuracy(str(output_model), test_images, test_labels)
    except Exception as e:
        print(f"Could not verify accuracy: {e}")


if __name__ == '__main__':
    main()
