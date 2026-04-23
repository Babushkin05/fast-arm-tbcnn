# Fast Ternary-Binary Neural Network Inference on ARM

Optimized inference runtime for ternary-binary neural networks on ARM processors with up to **23x speedup** for matrix operations and **18x speedup** for convolutional layers.

## Overview

This project implements a high-performance inference engine for neural networks with quantized weights (ternary: -1, 0, +1 or binary: -1, +1). It achieves significant speedups through:

- **Bit-packing**: 16x memory reduction for binary weights
- **Cache-optimized GeMM**: Blocked matrix multiplication tuned for ARM caches
- **im2col transformation**: Conv2D → optimized GeMM
- **ARM NEON SIMD**: Vectorized operations
- **Auto-quantization**: Float weights quantized at inference time

## Performance

### GeMM Speedup (Float × Binary vs Float × Float)

| Matrix Size | Float × Binary | Float × Float | Speedup |
|-------------|----------------|---------------|---------|
| 128×128 | 880 μs | 4.1 ms | **4.6x** |
| 256×256 | 3.9 ms | 31.0 ms | **8.0x** |
| 512×512 | 22.8 ms | 371.6 ms | **16.3x** |
| 1024×1024 | 133 ms | 3045 ms | **22.9x** |

### Conv2D Speedup (im2col + Binary GeMM)

| Layer | Naive | Optimized | Speedup |
|-------|-------|-----------|---------|
| 64×14×14 → 128×14×14 | 10425 μs | 577 μs | **18x** |

## Quick Start

### Build

```bash
# Build tbn-runtime
cd tbn-runtime
mkdir build && cd build
cmake ..
cmake --build . --target tbn

# Build test executable
cmake --build . --target test_onnx_inference
```

### Run CNN Inference

```bash
./bin/test_onnx_inference tests/models/cnn/onnx/simple_cnn.onnx
```

Output:
```
Model loaded successfully!
Input: input shape: [1, 1, 28, 28]

Running inference...
[INFO ] Conv2D binary (optimized): [1, 1, 28, 28] -> [1, 6, 24, 24]
[INFO ] MaxPool
[INFO ] Conv2D binary (optimized): [1, 6, 12, 12] -> [1, 16, 8, 8]
[INFO ] MaxPool
[INFO ] Gemm (Linear)

Inference completed!
Output: output shape: [1, 10]
```

## Project Structure

```
fast-arm-tbcnn/
├── tbn-runtime/           # Main inference runtime
│   ├── include/tbn/       # Public headers
│   │   ├── runtime/       # Model, Session, Tensor
│   │   ├── operators/     # Conv2D, Gemm, Pooling
│   │   ├── quantization/  # Quantization utilities
│   │   └── utils/         # Logging, errors
│   ├── src/               # Implementation
│   ├── examples/          # Usage examples
│   └── benchmarks/        # Performance tests
│
├── GeMM/                  # Optimized matrix multiplication
│   ├── 01-naive/          # Baseline
│   ├── 02-coded/          # Bit-packing
│   ├── 03-blocked/        # Cache-aware blocking
│   ├── 04-neon/           # NEON SIMD
│   ├── 05-final/          # Production API
│   └── bench/             # Benchmark suite
│
├── tests/                 # Test models
│   └── models/
│       ├── simple/        # Basic test models
│       └── cnn/           # CNN test models
│
└── ADR/                   # Architecture Decision Records
    ├── 0001-onnx-runtime-integration.md
    ├── 0002-quantization-strategy.md
    ├── 0003-memory-layout.md
    ├── 0005-error-handling-strategy.md
    ├── 0006-conv2d-optimization.md
    └── 0007-pooling-operations.md
```

## Supported Operators

| Operator | Status | Optimization |
|----------|--------|--------------|
| Conv2D | ✅ | im2col + Binary GeMM, auto-quantization |
| Gemm / MatMul | ✅ | Blocked cache-optimized, 1D bias support |
| MaxPool2D | ✅ | Naive (pooling rarely bottleneck) |
| AvgPool2D | ✅ | Naive |
| GlobalMaxPool | ✅ | Naive |
| GlobalAvgPool | ✅ | Naive |
| ReLU | ✅ | Element-wise |
| Add | ✅ | Broadcasting support |
| Reshape / Flatten | ✅ | Shape manipulation |

## Usage

### C++ API

```cpp
#include <tbn/tbn.hpp>

// Load ONNX model
auto model = tbn::load_onnx_model("model.onnx");

// Create inference session
auto session = model.create_session();

// Prepare input tensor
tbn::Shape input_shape{1, 1, 28, 28};
tbn::Tensor input(input_shape, tbn::DataType::FLOAT32);
// ... fill input data ...

// Run inference
session.set_input("input", input);
session.run();
auto output = session.get_output("output");

// Access results
float* output_data = output.typed_data<float>();
```

### Quantization

```cpp
#include <tbn/operators/quantized_gemm.hpp>

// Quantize float weights to binary
Tensor binary_weights = tbn::quantize_to_binary(float_weights);

// Optimized binary GeMM
Tensor result = tbn::qlinear_matmul_binary(input, binary_weights, scale);
```

## Architecture Decisions

Key architectural decisions are documented in the [ADR/](ADR/) directory:

- **ADR-0001**: Custom ONNX parser (not ONNX Runtime) for minimal overhead
- **ADR-0002**: Runtime quantization with on-the-fly conversion
- **ADR-0003**: Packed bit representation for weights
- **ADR-0006**: Conv2D via im2col + optimized GeMM
- **ADR-0007**: Naive pooling implementation

## Dependencies

- **C++17** compatible compiler (Clang, GCC)
- **CMake 3.16+**
- **ONNX** (for model loading, optional)
- **ARM NEON** (for SIMD, auto-detected)

## Tested Platforms

| Platform | CPU | Status |
|----------|-----|--------|
| macOS | Apple M4 Pro | ✅ |
| Raspberry Pi | Cortex-A72 | ✅ |
| Android | Snapdragon 720G | ✅ |

## References

1. Trusov et al. "Fast matrix multiplication for binary and ternary CNNs on ARM CPU" (2022)
2. Goto & van de Geijn "Anatomy of High-Performance Matrix Multiplication" (2008)

## License

MIT License - see [LICENSE](LICENSE) for details.
