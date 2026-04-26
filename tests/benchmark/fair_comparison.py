#!/usr/bin/env python3
"""
Fair comparison: TBN vs ONNX Runtime (single-threaded)
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add TBN Python module
tbn_build_dir = Path(__file__).parent.parent.parent / 'tbn-runtime' / 'build' / 'python'
sys.path.insert(0, str(tbn_build_dir))

import onnxruntime as ort
import tbn

def benchmark_onnx_single_thread(model_path, images, labels, warmup=10, runs=100):
    """ONNX Runtime with single thread for fair comparison."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(str(model_path), so, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # Warmup
    for i in range(warmup):
        _ = sess.run(None, {input_name: images[i % len(images)].reshape(1, 1, 28, 28).astype(np.float32)})

    # Benchmark
    latencies = []
    correct = 0

    for i in range(runs):
        img = images[i % len(images)].reshape(1, 1, 28, 28).astype(np.float32)

        start = time.perf_counter()
        output = sess.run(None, {input_name: img})[0]
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

        if np.argmax(output) == labels[i % len(labels)]:
            correct += 1

    return np.mean(latencies), np.std(latencies), correct / runs * 100

def benchmark_onnx_multi_thread(model_path, images, labels, warmup=10, runs=100):
    """ONNX Runtime with default threading (uses all cores)."""
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # Warmup
    for i in range(warmup):
        _ = sess.run(None, {input_name: images[i % len(images)].reshape(1, 1, 28, 28).astype(np.float32)})

    # Benchmark
    latencies = []
    correct = 0

    for i in range(runs):
        img = images[i % len(images)].reshape(1, 1, 28, 28).astype(np.float32)

        start = time.perf_counter()
        output = sess.run(None, {input_name: img})[0]
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

        if np.argmax(output) == labels[i % len(labels)]:
            correct += 1

    return np.mean(latencies), np.std(latencies), correct / runs * 100

def benchmark_tbn_float(model_path, images, labels, warmup=10, runs=100):
    """TBN with float weights."""
    model = tbn.load_model(str(model_path))

    # Warmup
    for i in range(warmup):
        _ = model.run(images[i % len(images)].reshape(1, 1, 28, 28))

    # Benchmark
    latencies = []
    correct = 0

    for i in range(runs):
        img = images[i % len(images)].reshape(1, 1, 28, 28)

        start = time.perf_counter()
        output = model.run(img)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

        if np.argmax(output) == labels[i % len(labels)]:
            correct += 1

    return np.mean(latencies), np.std(latencies), correct / runs * 100

def benchmark_tbn_quantized(model_path, images, labels, warmup=10, runs=100):
    """TBN with binary weight quantization."""
    model = tbn.load_model(str(model_path))

    # Warmup
    for i in range(warmup):
        _ = model.run_quantized(images[i % len(images)].reshape(1, 1, 28, 28))

    # Benchmark
    latencies = []
    correct = 0

    for i in range(runs):
        img = images[i % len(images)].reshape(1, 1, 28, 28)

        start = time.perf_counter()
        output = model.run_quantized(img)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

        if np.argmax(output) == labels[i % len(labels)]:
            correct += 1

    return np.mean(latencies), np.std(latencies), correct / runs * 100

def load_mnist_test_data(num_samples=100):
    """Load MNIST test data."""
    import kagglehub
    import gzip
    import struct
    import os

    print("Loading MNIST test data...")
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")

    images_file = os.path.join(path, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
    labels_file = os.path.join(path, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

    with open(images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with open(labels_file, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    images = images[:num_samples].astype(np.float32) / 255.0
    images = (images - MNIST_MEAN) / MNIST_STD
    labels = labels[:num_samples]

    return images, labels

def main():
    print("=" * 70)
    print("Fair Comparison: TBN vs ONNX Runtime (Single-Threaded)")
    print("=" * 70)

    # Load data
    images, labels = load_mnist_test_data(100)

    # Model path
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / 'train' / 'mnist' / 'mnist_tbn.onnx'

    print(f"\nModel: {model_path.name}")
    print(f"Test samples: {len(images)}")
    print()

    results = []

    # ONNX Runtime - Multi-threaded (default)
    print("[ONNX Runtime - Multi-threaded] Benchmarking...")
    mean, std, acc = benchmark_onnx_multi_thread(model_path, images, labels)
    results.append(('ONNX (multi-thread)', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # ONNX Runtime - Single-threaded
    print("[ONNX Runtime - Single-threaded] Benchmarking...")
    mean, std, acc = benchmark_onnx_single_thread(model_path, images, labels)
    results.append(('ONNX (single-thread)', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # TBN Float
    print("[TBN Float] Benchmarking...")
    mean, std, acc = benchmark_tbn_float(model_path, images, labels)
    results.append(('TBN (float)', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # TBN Quantized
    print("[TBN Quantized] Benchmarking...")
    mean, std, acc = benchmark_tbn_quantized(model_path, images, labels)
    results.append(('TBN (quantized)', mean, std, acc))
    print(f"  Latency: {mean:.3f} ± {std:.3f} ms, Accuracy: {acc:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Runtime':<25} {'Latency (ms)':<20} {'Accuracy'}")
    print("-" * 70)
    for name, mean, std, acc in results:
        print(f"{name:<25} {mean:.3f} ± {std:.3f}         {acc:.1f}%")

    # Speedup comparison
    onnx_single = results[1][1]
    tbn_quant = results[3][1]
    print("\n" + "=" * 70)
    print("Single-threaded Comparison:")
    print(f"  ONNX (1 thread):  {onnx_single:.3f} ms")
    print(f"  TBN (quantized):  {tbn_quant:.3f} ms")
    if tbn_quant < onnx_single:
        print(f"  TBN is {onnx_single/tbn_quant:.1f}x FASTER!")
    else:
        print(f"  ONNX is {tbn_quant/onnx_single:.1f}x faster")
    print("=" * 70)

if __name__ == '__main__':
    main()
