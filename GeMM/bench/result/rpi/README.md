# Benchmark Results

Ternary-Binary GeMM benchmark results on Raspberry Pi.

## Hardware

- **Device**: Raspberry Pi (4-core ARM Cortex)
- **L1 Data Cache**: 32 KB (per core)
- **L2 Cache**: 1 MB (shared)

## Tiling Parameters

Parameters calculated based on cache sizes using formulas from:
1. Trusov et al. "Fast matrix multiplication for binary and ternary CNNs on ARM CPU" (2022)
2. Goto & van de Geijn "Anatomy of High-Performance Matrix Multiplication" (2008)

```
mmk  = 16   (microkernel rows - fixed for ARMv8 NEON)
nmk  = 8    (microkernel cols - fixed for ARMv8 NEON)
kblk = 256  (depth block - from L1 constraint)
mblk = 128  (row block - from L2 constraint)
nblk = 256  (col block - from L2 constraint)
```

## Benchmark Command

```bash
for impl in 01 02 03 04 05; do
    ./bench_gemm_${impl} \
        --sizes "128,256,512" \
        --types "random_dense,random_sparse,dense_no_zero,diagonal,banded,block_sparse" \
        --runs 30 \
        --warmup 5 \
        --pause-ms 20 \
        --device "rpi" \
        --mblk 128 --nblk 256 --kblk 256 --mmk 16 --nmk 8 \
        --output "rpi_${impl}.csv"
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
| 01-naive       | 0.40       | 1.0×                |
| 02-coded       | 9.73       | 24.4×               |
| 03-blocked     | 9.29       | 23.3×               |
| 04-neon        | 9.28       | 23.3×               |
| **05-final**   | **12.46**  | **31.2×**           |

### GFLOPS by Matrix Size

| Size  | 01-naive | 02-coded | 03-blocked | 04-neon | 05-final |
|-------|----------|----------|------------|---------|----------|
| 128   | 0.38     | 8.68     | 5.91       | 5.94    | **14.23**|
| 256   | 0.41     | 6.31     | 7.58       | 7.60    | **8.45** |
| 512   | 0.40     | 14.20    | 14.38      | 14.29   | **14.70**|

### GFLOPS by Matrix Type

| Type           | 01-naive | 02-coded | 03-blocked | 04-neon | 05-final |
|----------------|----------|----------|------------|---------|----------|
| random_dense   | 0.40     | 10.00    | 10.63      | 10.57   | **13.67**|
| random_sparse  | 0.37     | 9.85     | 9.12       | 9.05    | **12.30**|
| dense_no_zero  | 0.41     | 9.78     | 9.09       | 9.05    | **12.31**|
| diagonal       | 0.44     | 9.97     | 8.95       | 8.99    | **12.17**|
| banded         | 0.42     | 9.94     | 8.96       | 8.97    | **12.14**|
| block_sparse   | 0.35     | 8.83     | 8.97       | 9.03    | **12.17**|

## Comparison with MacBook M4 Pro

| Implementation | RPi GFLOPS | M4 Pro GFLOPS | Ratio (M4/RPi) |
|----------------|------------|---------------|----------------|
| 01-naive       | 0.40       | 0.38          | 0.95×          |
| 02-coded       | 9.73       | 17.98         | 1.85×          |
| 03-blocked     | 9.29       | 24.57         | 2.64×          |
| 04-neon        | 9.28       | 25.09         | 2.70×          |
| 05-final       | 12.46      | 21.78         | 1.75×          |

**Key observations:**
- M4 Pro is ~2-3× faster for optimized implementations (03-04)
- 05-final shows best performance on RPi (31.2× speedup vs naive)
- 02-coded, 03-blocked, 04-neon show similar performance on RPi (~9-10 GFLOPS)
- Unlike M4 Pro where 04-neon is fastest, on RPi the 05-final wins

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

### 03-blocked
- Cache-aware blocking with tiling parameters
- 64-bit word processing with `std::popcount`
- Pack/unpack for cache locality

### 04-neon
- NEON SIMD for 128-bit operations
- Uses scalar popcount (NEON popcount not efficient on this CPU)

### 05-final (Fastest on RPi)
- Modern C++ API with namespaces and type safety
- Optimized memory management
- Best overall performance on Raspberry Pi

## Files

```
rpi/
├── README.md           # This file
├── rpi_01.csv          # 01-naive results (540 rows)
├── rpi_02.csv          # 02-coded results (540 rows)
├── rpi_03.csv          # 03-blocked results (540 rows)
├── rpi_04.csv          # 04-neon results (540 rows)
└── rpi_05.csv          # 05-final results (540 rows)
```

Each CSV contains 540 rows = 3 sizes × 6 types × 30 runs.

## CSV Format

```csv
device,impl,matrix_type,m,n,k,run,time_ms,gflops,mblk,nblk,kblk,mmk,nmk
rpi,05-final,random_dense,128,128,128,1,0.308643,13.589500,128,256,256,16,8
```

- `device`: Device identifier
- `impl`: Implementation name
- `matrix_type`: Type of test matrix
- `m,n,k`: Matrix dimensions (A is m×n, B is n×k, C is m×k)
- `run`: Run number (1-30)
- `time_ms`: Execution time in milliseconds
- `gflops`: Performance in GFLOPS (2×m×n×k operations)
- `mblk,nblk,kblk,mmk,nmk`: Tiling parameters used
