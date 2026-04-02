# Benchmark Results

Ternary-Binary GeMM benchmark results on Samsung Galaxy A52.

## Hardware

- **Device**: Samsung Galaxy A52 (SM-A525F)
- **CPU**: Qualcomm Snapdragon 720G (SM7125)
- **Cores**: 8 (2× Kryo 465 Gold @ 2.3 GHz + 6× Kryo 465 Silver @ 1.8 GHz)
- **L1 Data Cache**: 64 KB (per core)
- **L2 Cache**: ~512 KB (per cluster)

## Tiling Parameters

Parameters calculated based on cache sizes using formulas from:
1. Trusov et al. "Fast matrix multiplication for binary and ternary CNNs on ARM CPU" (2022)
2. Goto & van de Geijn "Anatomy of High-Performance Matrix Multiplication" (2008)

```
mmk  = 16   (microkernel rows - fixed for ARMv8 NEON)
nmk  = 8    (microkernel cols - fixed for ARMv8 NEON)
kblk = 256  (depth block - from L1 constraint)
mblk = 64   (row block - from L2 constraint)
nblk = 128  (col block - from L2 constraint)
```

## Benchmark Command

```bash
adb push build-android/bench/bench_gemm_* /data/local/tmp/
adb shell "cd /data/local/tmp && for impl in 01 02 03 04 05; do
    ./bench_gemm_\${impl} \\
        --sizes '128,256,512' \\
        --types 'random_dense,random_sparse,dense_no_zero,diagonal,banded,block_sparse' \\
        --runs 30 \\
        --warmup 5 \\
        --pause-ms 20 \\
        --device 'samsung-a52' \\
        --mblk 64 --nblk 128 --kblk 256 --mmk 16 --nmk 8 \\
        --output /sdcard/a52_\${impl}.csv
done"
adb pull /sdcard/a52_*.csv .
```

## Matrix Types

| Type | Ternary A Distribution | Use Case |
|------|------------------------|----------|
| `random_dense` | 33% each: -1, 0, +1 | Baseline, typical neural network |
| `random_sparse` | 80% zeros, 10% each ±1 | Pruned networks |
| `dense_no_zero` | 50/50 ±1, no zeros | Worst-case popcount |
| `diagonal` | Only diagonal nonzero | Minimal work, max cache locality |
| `banded` | Band width 8 around diagonal | Convolutional layers |
| `block_sparse` | 8×8 blocks, 30% dense | Structured pruning |

## Results Summary

### Average GFLOPS by Implementation

| Implementation | Avg GFLOPS | Speedup vs 01-naive |
|----------------|------------|---------------------|
| 01-naive       | 0.52       | 1.0×                |
| **02-coded**   | **15.03**  | **29.2×**           |
| 03-blocked     | 12.25      | 23.8×               |
| 04-neon        | 12.52      | 24.3×               |
| 05-final       | 12.77      | 24.8×               |

### GFLOPS by Matrix Size

| Size | 01-naive | 02-coded | 03-blocked | 04-neon | 05-final |
|------|----------|----------|------------|---------|----------|
| 128  | 0.31     | 8.03     | 8.06       | **8.46**  | 8.17     |
| 256  | 0.60     | **9.52** | 7.63       | 7.70    | 7.78     |
| 512  | 0.63     | **27.54**| 21.05      | 21.40   | 22.35    |

### GFLOPS by Matrix Type

| Type           | 01-naive | 02-coded | 03-blocked | 04-neon | 05-final |
|----------------|----------|----------|------------|---------|----------|
| random_dense   | 0.53     | **15.18**| 12.24      | 12.51   | 12.72    |
| random_sparse  | 0.52     | **15.16**| 12.26      | 12.51   | 12.82    |
| dense_no_zero  | 0.53     | **15.12**| 12.25      | 12.54   | 12.77    |
| diagonal       | 0.52     | **14.93**| 12.26      | 12.51   | 12.78    |
| banded         | 0.52     | **14.95**| 12.25      | 12.51   | 12.77    |
| block_sparse   | 0.51     | **14.86**| 12.23      | 12.53   | 12.77    |

## Key observations:

- **02-coded is fastest** on Samsung A52 with 15.03 GFLOPS average (29.2× speedup vs naive)
- Unlike M4 Pro and RPi, the simpler implementation (02-coded) outperforms blocked/NEON versions
- 03/04/05 implementations show similar performance (~12-13 GFLOPS)
- Performance scales dramatically with matrix size (27.5 GFLOPS at 512×512 for 02-coded)
- Matrix type has minimal impact on performance

## Comparison with Other Devices

| Implementation | A52 GFLOPS | RPi GFLOPS | M4 Pro GFLOPS |
|----------------|------------|------------|---------------|
| 01-naive       | 0.52       | 0.40       | 2.89          |
| **02-coded**   | **15.03**  | 9.73       | 69.43         |
| 03-blocked     | 12.25      | 9.29       | **137.99**    |
| 04-neon        | 12.52      | 9.28       | 137.62        |
| 05-final       | 12.77      | **12.46**  | 127.30        |

**Key insight**: Snapdragon 720G favors the simpler 02-coded implementation, possibly due to:
- Smaller cache sizes making blocking overhead more costly
- Different memory hierarchy characteristics
- Cache blocking overhead not justified for this CPU

## Implementation Notes

### 01-naive
- Simple O(m×n×k) triple loop
- Bit-by-bit decoding of ternary and binary values
- No SIMD, no blocking
- Baseline for comparison

### 02-coded (Fastest on A52)
- Uses `std::popcount` for hardware bit counting
- 64-bit word processing
- No cache blocking - simpler memory access pattern
- Minimal overhead works well on this CPU

### 03-blocked
- Cache-aware blocking with tiling parameters
- 64-bit word processing with `std::popcount`
- Pack/unpack overhead may hurt on smaller caches

### 04-neon
- NEON SIMD for 128-bit operations
- Uses scalar popcount

### 05-final
- Modern C++ API with namespaces and type safety
- Slight overhead from abstractions

## Files

```
a52/
├── README.md           # This file
├── a52_01.csv          # 01-naive results (540 rows)
├── a52_02.csv          # 02-coded results (540 rows)
├── a52_03.csv          # 03-blocked results (540 rows)
├── a52_04.csv          # 04-neon results (540 rows)
└── a52_05.csv          # 05-final results (540 rows)
```

Each CSV contains 540 rows = 3 sizes × 6 types × 30 runs.

## CSV Format

```csv
device,impl,matrix_type,m,n,k,run,time_ms,gflops,mblk,nblk,kblk,mmk,nmk
samsung-a52,02-coded,random_dense,512,512,512,1,19.8,27.5,64,128,256,16,8
```

- `device`: Device identifier
- `impl`: Implementation name
- `matrix_type`: Type of test matrix
- `m,n,k`: Matrix dimensions (A is m×n, B is n×k, C is m×k)
- `run`: Run number (1-30)
- `time_ms`: Execution time in milliseconds
- `gflops`: Performance in GFLOPS (2×m×n×k operations)
- `mblk,nblk,kblk,mmk,nmk`: Tiling parameters used
