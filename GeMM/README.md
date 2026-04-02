# Ternary-Binary GeMM

Optimized General Matrix Multiply (GeMM) implementations for ternary-binary neural networks using bit-packing and ARM NEON SIMD instructions.

## Implementations

| ID | Name | Description | Key Technique |
|----|------|-------------|---------------|
| 01 | naive | Baseline O(m×n×k) | Bit-by-bit decoding |
| 02 | coded | Bit-packing | Hardware popcount (64-bit) |
| 03 | blocked | Cache-aware | Tiling parameters |
| 04 | neon | SIMD | NEON 128-bit operations |
| 05 | final | Production | Modern C++ API |

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Quick Test

```bash
./build/bench/bench_gemm_05 --sizes 128 --device test --output results.csv
```

## Benchmark Results

See [bench/README.md](bench/README.md) for detailed results and analysis.

### Summary

| Device | CPU | Best Implementation | GFLOPS | Speedup |
|--------|-----|---------------------|--------|---------|
| MacBook M4 Pro | Apple M4 Pro | 03-blocked | 137.99 | 48× |
| Raspberry Pi | Cortex-A72 | 05-final | 12.46 | 31× |
| Samsung A52 | Snapdragon 720G | 02-coded | 15.03 | 29× |

### Device Comparison

![Device Comparison](bench/plots/figures/device_comparison.png)

### Performance by Matrix Size

![GFLOPS Comparison](bench/plots/figures/gflops_comparison.png)

### Time Distribution

![Time Distribution](bench/plots/figures/time_distribution.png)

## Key Findings

- **M4 Pro**: Large L2 cache (4 MB) benefits from blocking → 03-blocked wins with 138 GFLOPS
- **RPi**: Medium cache (1 MB) → 05-final with smart memory management
- **A52**: Small cache (512 KB) → simple 02-coded avoids blocking overhead

## API Usage (05-final)

```cpp
#include "GeMM.hpp"
using namespace tbn;

// Pack matrices
TernaryMatrix A = TernaryMatrix::pack(data_a, m, n);
BinaryMatrix B = BinaryMatrix::pack(data_b, n, k);

// Compute
GemmEngine engine;
Int32Matrix C = engine.compute(A.view(), B.view());

// Access results
int32_t value = C.at(row, col);
```

## Project Structure

```
GeMM/
├── 01-naive/          # Baseline implementation
├── 02-coded/          # Bit-packing with popcount
├── 03-blocked/        # Cache-aware blocking
├── 04-neon/           # NEON SIMD optimizations
├── 05-final/          # Production API
├── bench/             # Benchmark suite & results
│   ├── result/        # CSV results per device
│   │   ├── m4pro/
│   │   ├── rpi/
│   │   └── a52/
│   └── plots/         # Visualization
└── CMakeLists.txt
```

## References

1. Trusov et al. "Fast matrix multiplication for binary and ternary CNNs on ARM CPU" (2022)
2. Goto & van de Geijn "Anatomy of High-Performance Matrix Multiplication" (2008)
