# GeMM Benchmark Suite

Benchmark suite for ternary-binary GeMM implementations on ARM processors.

## Build

```bash
cd ../GeMM  # from bench directory
cmake -B build && cmake --build build
```

This produces four executables in `build/bench/`:
- `bench_gemm_02` — coded implementation (baseline)
- `bench_gemm_03` — blocked implementation
- `bench_gemm_04` — NEON-optimized
- `bench_gemm_05` — final API with NEON

## Usage

```bash
./bench_gemm_05 [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sizes <list>` | Matrix sizes, comma-separated (m=n=k) | *required* |
| `--types <list>` | Matrix types, comma-separated | `random_dense` |
| `--runs <n>` | Number of runs per configuration | 100 |
| `--warmup <n>` | Warmup runs before measurement | 5 |
| `--pause-ms <ms>` | Pause between runs (ms) | 50 |
| `--device <name>` | Device identifier for CSV | `unknown` |
| `--output <file>` | Output CSV filename | `results.csv` |
| `--mblk <n>` | Tiling: outer block rows | 64 |
| `--nblk <n>` | Tiling: outer block cols | 64 |
| `--kblk <n>` | Tiling: outer block depth | 128 |
| `--mmk <n>` | Tiling: microkernel rows | 32 |
| `--nmk <n>` | Tiling: microkernel cols | 32 |

### Matrix Types

| Type | Ternary A Distribution | Use Case |
|------|------------------------|----------|
| `random_dense` | 33% each: -1, 0, +1 | Baseline, typical neural network |
| `random_sparse` | 80% zeros, 10% each ±1 | Pruned networks |
| `dense_no_zero` | 50/50 ±1, no zeros | Worst-case popcount |
| `diagonal` | Only diagonal nonzero | Minimal work, max cache locality |
| `banded` | Band width 8 around diagonal | Convolutional layers |
| `block_sparse` | 8x8 blocks, 30% dense | Structured pruning |

### Examples

```bash
# Quick test
./bench_gemm_05 --sizes 128 --runs 10 --device test

# Full benchmark on MacBook
./bench_gemm_05 \
    --sizes "64,128,256,512" \
    --types "random_dense,random_sparse,block_sparse" \
    --runs 100 \
    --warmup 10 \
    --pause-ms 50 \
    --device "macbook-m1" \
    --output macbook_results.csv

# Compare implementations
./bench_gemm_02 --sizes 256 --device macbook-m1 --output impl02.csv
./bench_gemm_05 --sizes 256 --device macbook-m1 --output impl05.csv
```

## Output Format

CSV with one row per run:

```csv
device,impl,matrix_type,m,n,k,run,time_ms,gflops,mblk,nblk,kblk,mmk,nmk
macbook-m1,05-final,random_dense,128,128,128,1,1.024,4.096,64,64,128,32,32
```

## Cross-Compilation

### Android (Samsung)

```bash
# Set NDK path
export NDK=/path/to/android-ndk

# Configure and build
cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build-android

# Deploy and run
adb push build-android/bench/bench_gemm_05 /data/local/tmp/
adb shell chmod +x /data/local/tmp/bench_gemm_05
adb shell /data/local/tmp/bench_gemm_05 \
    --sizes "64,128,256" \
    --device "samsung-s21" \
    --output /sdcard/results.csv

# Retrieve results
adb pull /sdcard/results.csv
```

### Raspberry Pi

Option 1: Native compilation (recommended)
```bash
# On Raspberry Pi
git clone <repo>
cd GeMM
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bench/bench_gemm_05 --device "rpi4" --sizes "64,128,256"
```

Option 2: Cross-compilation
```bash
cmake -B build-rpi \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-rpi
# Copy build-rpi/bench/bench_gemm_05 to RPi
```

## Plotting

Python script for visualization:

```bash
cd plots
pip install -r requirements.txt
python plot_results.py ../results.csv --output-dir ./figures
```

Generates:
- `gflops_comparison.png` — performance by matrix size and type
- `time_distribution.png` — box plots of time distribution
- `device_comparison.png` — cross-device comparison (if multiple devices)
- `statistics.csv` — aggregated statistics (mean, std, min, max)

## Directory Structure

```
bench/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp           # Entry point, argument parsing
│   ├── matrix_gen.cpp     # Matrix generation
│   ├── matrix_gen.hpp
│   ├── runner.cpp         # Timing, GFLOPS calculation
│   ├── runner.hpp
│   ├── csv_writer.cpp     # CSV output
│   └── csv_writer.hpp
└── plots/
    ├── plot_results.py    # Matplotlib visualization
    └── requirements.txt   # Python dependencies
```
