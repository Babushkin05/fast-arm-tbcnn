# Ternary-Binary GeMM Benchmarks

Benchmark results for ternary-binary General Matrix Multiply (GeMM) implementations on ARM processors.

## Overview

This project implements optimized GeMM operations for ternary-binary neural networks using bit-packing and SIMD instructions. Five progressive implementations are benchmarked on three different ARM devices.

## Implementations

| ID | Name | Description |
|----|------|-------------|
| 01 | naive | Baseline O(m×n×k) with bit-by-bit decoding |
| 02 | coded | Bit-packing with hardware popcount (64-bit) |
| 03 | blocked | Cache-aware blocking with tiling parameters |
| 04 | neon | NEON SIMD optimizations (128-bit operations) |
| 05 | final | Production API with modern C++ design |

## Devices Tested

| Device | CPU | Cores | L1 Cache | L2 Cache |
|--------|-----|-------|----------|----------|
| MacBook M4 Pro | Apple M4 Pro | 10 | 128 KB/core | 4 MB |
| Raspberry Pi | Cortex-A72 | 4 | 32 KB/core | 1 MB |
| Samsung A52 | Snapdragon 720G | 8 | 64 KB/core | 512 KB |

## Results Summary

### Performance Comparison

![Device Comparison](plots/figures/device_comparison.png)

### GFLOPS by Matrix Size

![GFLOPS Comparison](plots/figures/gflops_comparison.png)

### Time Distribution

![Time Distribution](plots/figures/time_distribution.png)

### Average GFLOPS by Device

| Implementation | M4 Pro | RPi | A52 (big core) |
|----------------|--------|-----|----------------|
| 01-naive | 0.55 | — | 0.58 |
| 02-coded | 15.29 | — | 14.28 |
| 03-blocked | 94.56 | — | 12.06 |
| 04-neon | 104.70 | — | 11.32 |
| 05-final (opt tiling) | **159.55** | **2.50** | **32.79** |

*Average across 128×128, 256×256, 512×512 (M4 Pro includes 1024×1024). RPi: 01-04 marked "—" (not measured separately, only 05-final with direct SSH measurement). A52: big core (Kryo 465 Gold @ 2.3 GHz) via taskset 40, optimal tiling for 05-final.*

### Speedup vs Naive Baseline

| Implementation | M4 Pro | A52 (big core) |
|----------------|--------|----------------|
| 02-coded | 27.8× | 24.6× |
| 03-blocked | **171.9×** | 20.8× |
| 04-neon | 190.4× | 19.5× |
| 05-final (opt tiling) | **290.1×** | **56.5×** |

*RPi baseline (01-naive) not measured — likely <0.5 GFLOPS. A52 03-blocked and 04-neon have lower speedups than 02-coded due to blocking overhead on small L2 cache (512 KB) outweighing benefits at small/medium matrix sizes.*

## Key Findings

### 1. Device-Specific Optimal Implementations

| Device | Best Implementation | GFLOPS (peak) | Why? |
|--------|---------------------|---------------|------|
| **M4 Pro** | 05-final (opt) | 238.43 @ 1024×1024 | Large caches (4 MB L2) + Apple AMX/SIMD benefit from blocking |
| **RPi 4** | 05-final (opt) | 3.35 @ 512×512 | Limited L2 (1 MB) — modern tiling still helps (+21% over old tiling) |
| **A52** | 05-final (opt) | 41.73 @ 512×512 | 512 KB L2 — optimal tiling (mblk=128, nblk=512, kblk=512) gives +37-57% over default |

### 2. Cache Size Impact

- **Large caches (M4 Pro: 4 MB L2)** → Blocking and packing overhead pays off, achieving ~238 GFLOPS at 1024×1024
- **Medium caches (RPi 4: 1 MB L2)** → Tiling helps (+21% with optimal vs default), but absolute performance limited by Cortex-A72 core
- **Small caches (A52: 512 KB L2)** → Optimal tiling is critical — wrong tiling (e.g., mblk=256) causes L2 thrashing. Best tiling: mblk=128, nblk=512, kblk=512, mmk=64, nmk=32

### 3. A52 big.LITTLE Impact

Samsung A52 has 2× Kryo 465 Gold (big) + 6× Kryo 465 Silver (little). Without core pinning, benchmark variance is **2-3×** (kernel scheduler migrates between cores). All A52 results use `taskset 40` to pin to big core 6.

### 4. Matrix Size Scaling (05-final with optimal tiling)

| Size | M4 Pro | RPi 4 | A52 (big core) |
|------|--------|-------|----------------|
| 128 | 77.89 | 1.18 | 23.37 |
| 256 | 130.67 | 2.96 | 33.27 |
| 512 | 191.22 | 3.35 | 41.73 |
| 1024 | **238.43** | — | — |

*RPi old tiling: 1.15, 2.76, 2.77. A52 default tiling: 14.88, 23.21, 30.46. Tiling optimization gives +21% (RPi) and +37-57% (A52).*

## Directory Structure

```
bench/
├── result/
│   ├── m4pro/          # MacBook M4 Pro results
│   │   ├── README.md
│   │   └── m4pro_*.csv
│   ├── rpi/            # Raspberry Pi results
│   │   ├── README.md
│   │   └── rpi_*.csv
│   └── a52/            # Samsung A52 results
│       ├── README.md
│       └── a52_*.csv
├── plots/
│   ├── plot_results.py
│   └── figures/
│       ├── device_comparison.png
│       ├── gflops_comparison.png
│       ├── time_distribution.png
│       └── statistics.csv
└── README.md           # This file
```

## How to Reproduce

### Build

```bash
cd GeMM
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run Benchmarks

```bash
cd build/bench
for impl in 01 02 03 04 05; do
    ./bench_gemm_${impl} \
        --sizes "128,256,512" \
        --types "random_dense,random_sparse,dense_no_zero,diagonal,banded,block_sparse" \
        --runs 30 \
        --warmup 5 \
        --device "your-device" \
        --output "results_${impl}.csv"
done
```

### Generate Plots

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib
python3 bench/plots/plot_results.py combined_results.csv --output-dir bench/plots/figures
```

### Cross-compilation for Android

```bash
# Cross-compile with NDK
cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24
cmake --build build-android

# Push and run with big-core pinning (A52: Kryo 465 Gold = CPUs 6-7)
adb push build-android/bench/bench_gemm_05 /data/local/tmp/
adb shell "taskset 40 /data/local/tmp/bench_gemm_05 --device samsung-a52 \
    --sizes 128,256,512 --mblk 128 --nblk 512 --kblk 512 --mmk 64 --nmk 32 \
    --runs 20 --warmup 5 --pause-ms 200 --output /sdcard/a52_05.csv"
adb pull /sdcard/a52_05.csv
```

### Native Build on Termux (Android)

```bash
# One-time setup in Termux
pkg install -y python clang cmake make git ninja protobuf

# Build onnx from source
git clone --depth 1 https://github.com/onnx/onnx.git
cd onnx && cmake -B build -DCMAKE_INSTALL_PREFIX=$PREFIX && cmake --build build

# Build and run GeMM benchmarks
cd GeMM && cmake -B build && cmake --build build
taskset 40 ./build/bench/bench_gemm_05 --sizes 128,256,512 --device samsung-a52
```

## References

1. Trusov et al. "Fast matrix multiplication for binary and ternary CNNs on ARM CPU" (2022)
2. Goto & van de Geijn "Anatomy of High-Performance Matrix Multiplication" (2008)
