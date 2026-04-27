#!/usr/bin/env python3
"""
CIFAR-10 Benchmark: TBN vs ONNX Runtime

Compares inference performance with pre-quantized binary weights.
"""

import sys
import time
import numpy as np
import os
import kagglehub
from PIL import Image
from pathlib import Path

# Add TBN Python module
tbn_build_dir = Path(__file__).parent.parent.parent / 'tbn-runtime' / 'build' / 'python'
sys.path.insert(0, str(tbn_build_dir))

import onnxruntime as ort
import tbn
from tbn import quantize_onnx_model


def load_cifar10_test(num_samples=100):
    """Load CIFAR-10 test data from Kaggle (image folders structure)."""
    print("Downloading CIFAR-10 from Kaggle...")
    path = kagglehub.dataset_download("ayush1220/cifar10")
    print(f"Dataset path: {path}")

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

    print(f"Loaded {len(images)} test images")
    return np.array(images[:num_samples], dtype=np.float32), np.array(labels[:num_samples])


def benchmark_onnx(model_path, images, labels, warmup=10, runs=100):
    """ONNX Runtime benchmark."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(str(model_path), so, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # Warmup
    for i in range(warmup):
        img = np.expand_dims(images[i % len(images)], axis=0)
        _ = sess.run(None, {input_name: img})

    # Benchmark
    latencies = []
    correct = 0

    for i in range(runs):
        img = np.expand_dims(images[i % len(images)], axis=0)

        start = time.perf_counter()
        output = sess.run(None, {input_name: img})[0]
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

        if np.argmax(output) == labels[i % len(labels)]:
            correct += 1

    return np.mean(latencies), np.std(latencies), correct / runs * 100


def benchmark_tbn(model_path, images, labels, warmup=10, runs=100, quantized=False):
    """TBN benchmark."""
    model = tbn.load_model(str(model_path))

    # Warmup
    for i in range(warmup):
        img = np.expand_dims(images[i % len(images)], axis=0)
        if quantized:
            _ = model.run_quantized(img)
        else:
            _ = model.run(img)

    # Benchmark
    latencies = []
    correct = 0

    for i in range(runs):
        img = np.expand_dims(images[i % len(images)], axis=0)

        start = time.perf_counter()
        if quantized:
            output = model.run_quantized(img)
        else:
            output = model.run(img)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

        if np.argmax(output) == labels[i % len(labels)]:
            correct += 1

    return np.mean(latencies), np.std(latencies), correct / runs * 100


def main():
    print("=" * 70)
    print("CIFAR-10 Benchmark: TBN vs ONNX Runtime")
    print("=" * 70)

    # Paths
    script_dir = Path(__file__).parent
    train_dir = script_dir.parent / 'train' / 'cifar10'
    original_model = train_dir / 'cifar10_tbn.onnx'
    binary_model = train_dir / 'cifar10_binary_weights.onnx'

    # Check original model exists
    if not original_model.exists():
        print(f"\nModel not found at {original_model}")
        print("Please run: python train_tbn_cifar10.py --epochs 20")
        return

    # Quantize model if not already done
    if not binary_model.exists():
        print("\n" + "=" * 70)
        print("QUANTIZING MODEL (one-time operation)")
        print("=" * 70)
        quantize_onnx_model(str(original_model), str(binary_model), weight_type='binary')
    else:
        print(f"\nUsing existing quantized model: {binary_model}")

    # Load data
    print("\nLoading CIFAR-10 test data...")
    images, labels = load_cifar10_test(100)
    print(f"Image shape: {images[0].shape}")

    # Model info
    print(f"\nOriginal model: {original_model.name} ({original_model.stat().st_size / 1024:.1f} KB)")
    print(f"Binary model: {binary_model.name} ({binary_model.stat().st_size / 1024:.1f} KB)")

    results = []

    # ONNX Runtime with original model (single-thread)
    print("\n[ONNX Runtime - Original] Benchmarking...")
    mean, std, acc = benchmark_onnx(original_model, images, labels)
    results.append(('ONNX Original', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # ONNX Runtime with binary model
    print("\n[ONNX Runtime - Binary] Benchmarking...")
    mean, std, acc = benchmark_onnx(binary_model, images, labels)
    results.append(('ONNX Binary', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # TBN Float (original model, no quantization)
    print("\n[TBN Float] Benchmarking...")
    mean, std, acc = benchmark_tbn(original_model, images, labels, quantized=False)
    results.append(('TBN (float)', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # TBN Quantized with original model (on-the-fly quantization)
    print("\n[TBN Quantized - On-the-fly] Benchmarking...")
    mean, std, acc = benchmark_tbn(original_model, images, labels, quantized=True)
    results.append(('TBN (on-the-fly)', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # TBN Quantized with pre-quantized binary model
    print("\n[TBN Quantized - Pre-quantized] Benchmarking...")
    mean, std, acc = benchmark_tbn(binary_model, images, labels, quantized=True)
    results.append(('TBN (pre-quant)', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Runtime':<25} {'Latency (ms)':<20} {'Accuracy'}")
    print("-" * 70)
    for name, mean, std, acc in results:
        print(f"{name:<25} {mean:.3f} ± {std:.3f}         {acc:.1f}%")

    # Performance comparison
    onnx_time = results[0][1]  # ONNX Original
    tbn_prequant_time = results[4][1]  # TBN pre-quantized

    print("\n" + "=" * 70)
    print("Performance Comparison (Pre-quantized TBN vs ONNX):")
    print(f"  ONNX (original):     {onnx_time:.3f} ms")
    print(f"  TBN (pre-quantized): {tbn_prequant_time:.3f} ms")

    if tbn_prequant_time < onnx_time:
        print(f"  TBN is {onnx_time/tbn_prequant_time:.1f}x FASTER!")
    else:
        print(f"  ONNX is {tbn_prequant_time/onnx_time:.1f}x faster")
    print("=" * 70)

    # Show improvement from pre-quantization
    tbn_onthefly_time = results[3][1]
    improvement = (tbn_onthefly_time - tbn_prequant_time) / tbn_onthefly_time * 100
    print(f"\nPre-quantization speedup: {improvement:.1f}% faster than on-the-fly")


if __name__ == '__main__':
    main()
