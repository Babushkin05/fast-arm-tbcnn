#!/usr/bin/env python3
"""
Direct GeMM performance comparison:
- TBN optimized GeMM (ternary x binary with NEON popcount)
- NumPy float GeMM
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add TBN Python module
tbn_build_dir = Path(__file__).parent.parent.parent / 'tbn-runtime' / 'build' / 'python'
sys.path.insert(0, str(tbn_build_dir))

def benchmark_tbn_gemm(m, n, k, warmup=5, runs=50):
    """Benchmark TBN Float x Binary GeMM."""
    import tbn

    # Create random float matrix A
    A = np.random.randn(m, k).astype(np.float32)

    # Create binary matrix B (values -1 or +1)
    B_float = np.random.choice([-1.0, 1.0], size=(k, n)).astype(np.float32)

    # For TBN, we need to create tensors
    # But the Python bindings only expose model inference, not raw GeMM
    # So we use a workaround - measure through a simple model

    return None  # Not directly accessible from Python

def benchmark_numpy_gemm(m, n, k, warmup=5, runs=50):
    """Benchmark NumPy float GeMM."""
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = A @ B

    # Benchmark
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = A @ B
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return np.mean(latencies), np.std(latencies)

def main():
    print("=" * 60)
    print("Direct GeMM Performance Comparison")
    print("=" * 60)

    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    print("\nNumPy Float x Float GeMM (for reference):")
    print(f"{'Size':<20} {'Latency (ms)':<15} {'GFLOPS'}")
    print("-" * 50)

    for m, n, k in sizes:
        mean, std = benchmark_numpy_gemm(m, n, k)
        gflops = 2.0 * m * n * k / (mean / 1000) / 1e9
        print(f"{m}x{n}x{k:<12} {mean:.3f} ± {std:.3f}    {gflops:.1f}")

    print("\n" + "=" * 60)
    print("TBN Float x Binary GeMM (NEON-optimized with popcount):")
    print("Run C++ benchmark: ./build/bin/benchmark_quantized_gemm")
    print("=" * 60)

if __name__ == '__main__':
    main()
