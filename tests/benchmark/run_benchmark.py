#!/usr/bin/env python3
"""
TBN Benchmark Suite

Compares inference performance across:
- TBN (float) - our runtime without quantization
- TBN (quantized) - ternary activations × binary weights
- ONNX Runtime - standard ONNX inference
- PyTorch - native eager mode

Metrics:
- Latency (ms per image)
- Throughput (images/sec)
- Model size (KB)
- Accuracy (% on MNIST test set)

Devices: M4 Pro, Samsung A52, Raspberry Pi 4
"""

import sys
import os
import time
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

# Add paths
script_dir = Path(__file__).parent.absolute()
tbn_build_dir = script_dir.parent.parent / 'tbn-runtime' / 'build' / 'python'
sys.path.insert(0, str(tbn_build_dir))

import numpy as np

# Results storage
results = []

def get_device_info():
    """Get device information."""
    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python': platform.python_version(),
    }

    # Try to get more specific device info
    if platform.system() == 'Darwin':
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                    capture_output=True, text=True)
            info['cpu'] = result.stdout.strip()
        except:
            info['cpu'] = 'Unknown Apple Silicon'

    return info


def load_mnist_test_data(num_samples=1000):
    """Load MNIST test data."""
    import kagglehub

    print("Loading MNIST test data...")
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")

    import gzip
    import struct

    images_file = os.path.join(path, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
    labels_file = os.path.join(path, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

    with open(images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with open(labels_file, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Normalize
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    images = images[:num_samples].astype(np.float32) / 255.0
    images = (images - MNIST_MEAN) / MNIST_STD
    labels = labels[:num_samples]

    print(f"Loaded {len(images)} test images")
    return images, labels


def benchmark_tbn_float(images, labels, model_path, warmup=10, runs=100):
    """Benchmark TBN with float weights (no quantization)."""
    import tbn

    print("\n[TBN Float] Loading model...")
    model = tbn.load_model(str(model_path))

    # Warmup
    print(f"[TBN Float] Warmup ({warmup} runs)...")
    for i in range(warmup):
        _ = model.run(images[i % len(images)].reshape(1, 1, 28, 28))

    # Benchmark latency
    print(f"[TBN Float] Benchmarking latency ({runs} runs)...")
    latencies = []
    correct = 0

    for i in range(runs):
        img = images[i % len(images)].reshape(1, 1, 28, 28)

        start = time.perf_counter()
        output = model.run(img)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

        if np.argmax(output) == labels[i % len(labels)]:
            correct += 1

    latency_mean = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput = 1000.0 / latency_mean  # images/sec
    accuracy = correct / runs * 100

    print(f"[TBN Float] Latency: {latency_mean:.3f} ± {latency_std:.3f} ms")
    print(f"[TBN Float] Throughput: {throughput:.1f} img/s")
    print(f"[TBN Float] Accuracy: {accuracy:.2f}%")

    return {
        'runtime': 'TBN (float)',
        'latency_ms': latency_mean,
        'latency_std_ms': latency_std,
        'throughput_ips': throughput,
        'accuracy_pct': accuracy,
    }


def benchmark_tbn_quantized(images, labels, model_path, warmup=10, runs=100):
    """Benchmark TBN with binary weights (quantized path)."""
    import tbn

    print("\n[TBN Quantized] Loading model...")
    model = tbn.load_model(str(model_path))

    # Warmup
    print(f"[TBN Quantized] Warmup ({warmup} runs)...")
    for i in range(warmup):
        _ = model.run_quantized(images[i % len(images)].reshape(1, 1, 28, 28))

    # Benchmark
    print(f"[TBN Quantized] Benchmarking ({runs} runs)...")
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

    latency_mean = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput = 1000.0 / latency_mean
    accuracy = correct / runs * 100

    print(f"[TBN Quantized] Latency: {latency_mean:.3f} ± {latency_std:.3f} ms")
    print(f"[TBN Quantized] Throughput: {throughput:.1f} img/s")
    print(f"[TBN Quantized] Accuracy: {accuracy:.2f}%")

    return {
        'runtime': 'TBN (quantized)',
        'latency_ms': latency_mean,
        'latency_std_ms': latency_std,
        'throughput_ips': throughput,
        'accuracy_pct': accuracy,
    }


def benchmark_onnx_runtime(images, labels, model_path, warmup=10, runs=100):
    """Benchmark ONNX Runtime."""
    import onnxruntime as ort

    print("\n[ONNX Runtime] Loading model...")
    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name

    # Warmup
    print(f"[ONNX Runtime] Warmup ({warmup} runs)...")
    for i in range(warmup):
        _ = sess.run(None, {input_name: images[i % len(images)].reshape(1, 1, 28, 28).astype(np.float32)})

    # Benchmark
    print(f"[ONNX Runtime] Benchmarking ({runs} runs)...")
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

    latency_mean = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput = 1000.0 / latency_mean
    accuracy = correct / runs * 100

    print(f"[ONNX Runtime] Latency: {latency_mean:.3f} ± {latency_std:.3f} ms")
    print(f"[ONNX Runtime] Throughput: {throughput:.1f} img/s")
    print(f"[ONNX Runtime] Accuracy: {accuracy:.2f}%")

    return {
        'runtime': 'ONNX Runtime',
        'latency_ms': latency_mean,
        'latency_std_ms': latency_std,
        'throughput_ips': throughput,
        'accuracy_pct': accuracy,
    }


def benchmark_pytorch(images, labels, model_path, warmup=10, runs=100):
    """Benchmark PyTorch native inference."""
    import torch
    import torch.nn as nn
    import onnx
    from onnx import numpy_helper

    print("\n[PyTorch] Loading model...")

    # Load ONNX and convert to PyTorch
    onnx_model = onnx.load(str(model_path))
    weights = {}
    for init in onnx_model.graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)

    # Build equivalent PyTorch model for TBN architecture
    # conv1: [16, 1, 5, 5], conv2: [32, 16, 5, 5], fc: [10, 512]
    # Pool1: 2x2 stride 2, Pool2: 3x3 stride 3
    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
            self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))      # [1, 16, 28, 28]
            x = torch.max_pool2d(x, 2, 2)      # [1, 16, 14, 14]
            x = torch.relu(self.conv2(x))      # [1, 32, 14, 14]
            x = torch.max_pool2d(x, 3, 3)      # [1, 32, 4, 4]
            x = x.view(x.size(0), -1)          # [1, 512]
            x = self.fc(x)                     # [1, 10]
            return x

    net = MNISTNet()

    # Load weights from ONNX
    net.conv1.weight.data = torch.from_numpy(weights['conv1_w'])
    net.conv1.bias.data = torch.from_numpy(weights['conv1_b'].flatten())
    net.conv2.weight.data = torch.from_numpy(weights['conv2_w'])
    net.conv2.bias.data = torch.from_numpy(weights['conv2_b'].flatten())
    # fc_w is stored as [10, 512], need to transpose for PyTorch Linear
    net.fc.weight.data = torch.from_numpy(weights['fc_w'])
    net.fc.bias.data = torch.from_numpy(weights['fc_b'].flatten())

    net.eval()

    # Warmup
    print(f"[PyTorch] Warmup ({warmup} runs)...")
    with torch.no_grad():
        for i in range(warmup):
            img = torch.from_numpy(images[i % len(images)].reshape(1, 1, 28, 28))
            _ = net(img)

    # Benchmark
    print(f"[PyTorch] Benchmarking ({runs} runs)...")
    latencies = []
    correct = 0

    with torch.no_grad():
        for i in range(runs):
            img = torch.from_numpy(images[i % len(images)].reshape(1, 1, 28, 28))

            start = time.perf_counter()
            output = net(img)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

            if output.argmax().item() == labels[i % len(labels)]:
                correct += 1

    latency_mean = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput = 1000.0 / latency_mean
    accuracy = correct / runs * 100

    print(f"[PyTorch] Latency: {latency_mean:.3f} ± {latency_std:.3f} ms")
    print(f"[PyTorch] Throughput: {throughput:.1f} img/s")
    print(f"[PyTorch] Accuracy: {accuracy:.2f}%")

    return {
        'runtime': 'PyTorch',
        'latency_ms': latency_mean,
        'latency_std_ms': latency_std,
        'throughput_ips': throughput,
        'accuracy_pct': accuracy,
    }


def get_model_size(model_path):
    """Get model file size in KB."""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / 1024


def run_benchmark(output_dir=None, num_samples=1000, warmup=10, runs=100):
    """Run all benchmarks."""
    print("=" * 60)
    print("TBN Benchmark Suite")
    print("=" * 60)

    # Device info
    device_info = get_device_info()
    print(f"\nDevice: {device_info.get('cpu', device_info['machine'])}")
    print(f"Platform: {device_info['platform']} {device_info['machine']}")

    # Paths - use TBN-trained model for fair comparison
    model_path = script_dir.parent / 'train' / 'mnist' / 'mnist_tbn.onnx'

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load data
    images, labels = load_mnist_test_data(num_samples)

    # Model size
    model_size = get_model_size(model_path)
    print(f"\nModel size: {model_size:.1f} KB")

    # Run benchmarks
    results = []

    # TBN Float
    try:
        r = benchmark_tbn_float(images, labels, model_path, warmup, runs)
        r['model_size_kb'] = model_size
        results.append(r)
    except Exception as e:
        print(f"[TBN Float] Error: {e}")

    # TBN Quantized (binary weights)
    try:
        r = benchmark_tbn_quantized(images, labels, model_path, warmup, runs)
        r['model_size_kb'] = model_size  # Same file size, but runtime uses less memory
        results.append(r)
    except Exception as e:
        print(f"[TBN Quantized] Error: {e}")

    # ONNX Runtime
    try:
        r = benchmark_onnx_runtime(images, labels, model_path, warmup, runs)
        r['model_size_kb'] = model_size
        results.append(r)
    except Exception as e:
        print(f"[ONNX Runtime] Error: {e}")

    # PyTorch
    try:
        r = benchmark_pytorch(images, labels, model_path, warmup, runs)
        r['model_size_kb'] = model_size
        results.append(r)
    except Exception as e:
        print(f"[PyTorch] Error: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Runtime':<20} {'Latency (ms)':<15} {'Throughput':<12} {'Accuracy'}")
    print("-" * 60)
    for r in results:
        print(f"{r['runtime']:<20} {r['latency_ms']:.3f} ± {r['latency_std_ms']:.3f}    {r['throughput_ips']:.1f} img/s  {r['accuracy_pct']:.2f}%")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            'timestamp': datetime.now().isoformat(),
            'device': device_info,
            'config': {
                'num_samples': num_samples,
                'warmup': warmup,
                'runs': runs,
            },
            'results': results,
        }

        output_file = output_dir / f"benchmark_{device_info['machine']}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TBN Benchmark Suite')
    parser.add_argument('--output', '-o', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--samples', '-n', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--warmup', '-w', type=int, default=10,
                        help='Number of warmup runs')
    parser.add_argument('--runs', '-r', type=int, default=100,
                        help='Number of benchmark runs')

    args = parser.parse_args()

    output_dir = script_dir / args.output
    run_benchmark(output_dir, args.samples, args.warmup, args.runs)
