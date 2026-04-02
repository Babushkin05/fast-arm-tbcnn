# Benchmark Results

Ternary-Binary GeMM benchmark results on Apple MacBook M4 Pro.

## Hardware

- **Device**: MacBook Pro (14-inch, 2024)
- **CPU**: Apple M4 Pro
- **Performance cores**: 10
- **L1 Data Cache**: 128 KB (per core)
- **L2 Cache**: 4 MB

## Tiling Parameters

Parameters calculated based on cache sizes using formulas from:
1. Trusov et al. "Fast matrix multiplication for binary and ternary CNNs on ARM CPU" (2022)
2. Goto & van de Geijn "Anatomy of High-Performance Matrix Multiplication" (2008)

```
mmk  = 16   (microkernel rows - fixed for ARMv8 NEON)
nmk  = 8    (microkernel cols - fixed for ARMv8 NEON)
kblk = 512  (depth block - from L1 constraint, capped)
mblk = 256  (row block - from L2 constraint)
nblk = 512  (col block - from L2 constraint)
```

## Benchmark Command

```bash
for impl in 01 02 03 04 05; do
    ../build/bench/bench_gemm_${impl} \
        --sizes "128,256,512,1024" \
        --types "random_dense,random_sparse,dense_no_zero,diagonal,banded,block_sparse" \
        --runs 30 \
        --warmup 5 \
        --pause-ms 20 \
        --device "macbook-m4-pro" \
        --mblk 256 --nblk 512 --kblk 512 --mmk 16 --nmk 8 \
        --output "m4pro_${impl}.csv"
done
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
| 01-naive       | 2.89       | 1.0×                |
| 02-coded       | 69.43      | 24.0×               |
| **03-blocked** | **137.99** | **47.7×**           |
| 04-neon        | 137.62     | 47.6×               |
| 05-final       | 127.30     | 44.0×               |

### GFLOPS by Matrix Size

| Size  | 01-naive | 02-coded | 03-blocked | 04-neon | 05-final |
|-------|----------|----------|------------|---------|----------|
| 128   | 1.73     | 18.34    | 62.07      | 61.45   | 55.44    |
| 256   | 2.86     | 30.94    | 81.69      | 77.47   | 67.35    |
| 512   | 3.49     | 75.03    | 170.66     | **173.38** | 150.79 |
| 1024  | 3.49     | 153.40   | **237.53** | 238.18 | 235.63   |

### GFLOPS by Matrix Type

| Type           | 01-naive | 02-coded | 03-blocked | 04-neon | 05-final |
|----------------|----------|----------|------------|---------|----------|
| random_dense   | 3.16     | 77.13    | **149.92** | 141.32  | 135.82   |
| random_sparse  | 2.99     | 73.53    | 144.34     | **150.72** | 148.16 |
| dense_no_zero  | 2.78     | 68.78    | **148.30** | 129.96  | 123.27   |
| diagonal       | 2.79     | 65.62    | **132.31** | 128.25  | 113.93   |
| banded         | 2.80     | 65.80    | **125.19** | 122.06  | 119.54   |
| block_sparse   | 2.85     | 65.71    | 127.87     | **153.40** | 123.09 |

## Key observations:

- **03-blocked and 04-neon are nearly identical** in performance (~138 GFLOPS average)
- Massive **47.7× speedup** with optimized implementations vs naive
- Performance scales dramatically with matrix size (up to 238 GFLOPS at 1024×1024)
- Matrix type impacts performance more than on other devices (~15% variation)
- 05-final has ~10% overhead from modern C++ abstractions

## Comparison with Other Devices

| Implementation | M4 Pro GFLOPS | RPi GFLOPS | A52 GFLOPS |
|----------------|---------------|------------|------------|
| 01-naive       | 2.89          | 0.40       | 0.52       |
| 02-coded       | 69.43         | 9.73       | **15.03**  |
| 03-blocked     | **137.99**    | 9.29       | 12.25      |
| 04-neon        | 137.62        | 9.28       | 12.52      |
| 05-final       | 127.30        | **12.46**  | 12.77      |

**Key insight**: M4 Pro achieves ~10× higher performance than RPi and A52 for optimized implementations,
demonstrating the benefits of larger caches and higher clock speeds for blocked algorithms.

## Implementation Notes

### 01-naive
- Simple O(m×n×k) triple loop
- Bit-by-bit decoding of ternary and binary values
- No SIMD, no blocking
- Baseline for comparison

### 02-coded
- Uses `std::popcount` for hardware bit counting
- 64-bit word processing
- No cache blocking

### 03-blocked (Fastest)
- Cache-aware blocking with tiling parameters
- 64-bit word processing with `std::popcount`
- Pack/unpack for cache locality
- Best overall performance on M4 Pro

### 04-neon
- NEON SIMD for 128-bit operations
- Uses scalar popcount on Apple Silicon
- Performance nearly identical to 03-blocked

### 05-final
- Modern C++ API with namespaces and type safety
- Same core algorithm as 04-neon
- ~10% overhead from abstractions

## Files

```
result/
├── README.md           # This file
├── m4pro_01.csv        # 01-naive results (720 rows)
├── m4pro_02.csv        # 02-coded results (720 rows)
├── m4pro_03.csv        # 03-blocked results (720 rows)
├── m4pro_04.csv        # 04-neon results (720 rows)
└── m4pro_05.csv        # 05-final results (720 rows)
```

Each CSV contains 720 rows = 4 sizes × 6 types × 30 runs.

## CSV Format

```csv
device,impl,matrix_type,m,n,k,run,time_ms,gflops,mblk,nblk,kblk,mmk,nmk
macbook-m4-pro,03-blocked,random_dense,512,512,512,1,4.78,140.2,256,512,512,16,8
```

- `device`: Device identifier
- `impl`: Implementation name
- `matrix_type`: Type of test matrix
- `m,n,k`: Matrix dimensions (A is m×n, B is n×k, C is m×k)
- `run`: Run number (1-30)
- `time_ms`: Execution time in milliseconds
- `gflops`: Performance in GFLOPS (2×m×n×k operations)
- `mblk,nblk,kblk,mmk,nmk`: Tiling parameters used
