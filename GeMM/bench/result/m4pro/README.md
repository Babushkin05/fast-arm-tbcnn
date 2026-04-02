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

Calculated via:
```bash
../../.local/calc_tiling.sh
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
| 01-naive       | 0.38       | 1.0×                |
| 02-coded       | 17.98      | 47.3×               |
| 03-blocked     | 24.57      | 64.7×               |
| **04-neon**    | **25.09**  | **66.0×**           |
| 05-final       | 21.78      | 57.3×               |

### GFLOPS by Matrix Size

| Size  | 01-naive | 02-coded | 03-blocked | 04-neon | 05-final |
|-------|----------|----------|------------|---------|----------|
| 128   | 0.35     | 13.02    | 21.16      | **21.32** | 16.98    |
| 256   | 0.40     | 17.37    | 24.35      | **25.06** | 20.89    |
| 512   | 0.40     | 20.29    | 26.50      | **26.97** | 24.08    |
| 1024  | 0.39     | 21.22    | 26.28      | **26.98** | 25.19    |

### GFLOPS by Matrix Type

| Type           | 01-naive | 02-coded | 03-blocked | 04-neon | 05-final |
|----------------|----------|----------|------------|---------|----------|
| random_dense   | 0.40     | 18.02    | 24.53      | **24.88** | 21.72    |
| random_sparse  | 0.35     | 18.09    | 24.54      | **25.19** | 21.73    |
| dense_no_zero  | 0.40     | 17.84    | 24.60      | **25.03** | 21.66    |
| diagonal       | 0.40     | 18.03    | 24.57      | **25.23** | 21.87    |
| banded         | 0.38     | 18.06    | 24.63      | **25.13** | 21.83    |
| block_sparse   | 0.36     | 17.81    | 24.57      | **25.05** | 21.90    |

## Key observations:

- **04-neon is fastest** on M4 Pro with 25.09 GFLOPS average (66× speedup vs naive)
- 03-blocked and 04-neon show similar performance (~25 GFLOPS)
- 05-final has slight overhead from abstractions (~4 GFLOPS slower than 04-neon)
- Performance scales well with matrix size (26.98 GFLOPS at 1024×1024)
- Matrix type has minimal impact on performance (~1-2% variation)
- Scalar popcount on Apple Silicon outperforms NEON SIMD for this workload

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

### 04-neon (Fastest)
- Originally used NEON SIMD for 128-bit operations
- **Optimized**: Replaced NEON popcount with scalar `std::popcount`
- On Apple Silicon, scalar 64-bit operations with hardware popcount
  are faster than NEON due to:
  - Direct GPR loads (no NEON-GPR transfer overhead)
  - Single-instruction hardware popcount
  - No memory round-trips

### 05-final
- Modern C++ API with namespaces and type safety
- Same core algorithm as 04-neon
- Slight overhead from abstractions (std::vector vs raw arrays)

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
macbook-m4-pro,04-neon,random_dense,512,512,512,1,19.8,13.6,256,512,512,16,8
```

- `device`: Device identifier
- `impl`: Implementation name
- `matrix_type`: Type of test matrix
- `m,n,k`: Matrix dimensions (A is m×n, B is n×k, C is m×k)
- `run`: Run number (1-30)
- `time_ms`: Execution time in milliseconds
- `gflops`: Performance in GFLOPS (2×m×n×k operations)
- `mblk,nblk,kblk,mmk,nmk`: Tiling parameters used
