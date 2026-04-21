# TBN Runtime

Ternary-Binary Network (TBN) Runtime - optimized inference runtime for ternary and binary neural networks on ARM processors.

## Overview

This runtime provides efficient execution of neural networks with ternary (-1, 0, +1) and binary (0, 1) weights on ARM processors using bit-packing and SIMD optimizations.

## Features

- Efficient ternary/binary weight storage (16x memory reduction)
- ARM NEON SIMD optimizations
- ONNX Runtime integration
- Per-channel quantization support
- Multi-threading support
- Cross-platform compatibility

## Project Structure

```
tbn-runtime/
├── include/tbn/              # Public headers
│   ├── runtime/              # Core runtime (Model, Session, Tensor)
│   ├── quantization/         # Quantization strategies
│   ├── operators/            # Ternary operators (Conv2D, Gemm, etc)
│   ├── memory/               # Memory management and layouts
│   └── utils/                # Utilities (logging, errors, profiling)
├── src/                      # Implementation
│   ├── runtime/              # Core runtime implementation
│   ├── quantization/         # Quantization implementation
│   ├── operators/            # Operator implementations
│   └── onnx_integration/     # ONNX Runtime integration
├── benchmarks/               # Performance benchmarks
├── tests/                    # Unit tests
├── examples/                 # Usage examples
└── third_party/              # External dependencies
```

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```cpp
#include <tbn/tbn.hpp>

// Load model
auto model = tbn::load_model("model.onnx");

// Create inference session
auto session = model.create_session();

// Run inference
session.set_input("input", input_tensor);
session.run();
auto output = session.get_output("output");
```

## Dependencies

- C++17 compatible compiler
- CMake 3.16+
- Threads support
- (Optional) ONNX Runtime for model loading

## License

[License type to be determined]

## Contributing

[Contribution guidelines to be added]