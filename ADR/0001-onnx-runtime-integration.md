# ADR-0001: ONNX Runtime Integration

Date: 2026-04-21

## Status

Implemented (2026-04-24)

## Context

For the ternary-binary neural network, we need a way to load and execute models from standard frameworks (PyTorch, TensorFlow). The solution must:
- Support standard model formats
- Enable replacement of standard operations with custom ternary implementations
- Require minimal changes to the model export process
- Provide good performance on ARM processors

## Decision

Implemented a custom ONNX parser (not ONNX Runtime) for minimal overhead and full control.

### Implementation Details

**Parser** (`src/onnx_integration/onnx_parser.cpp`):
- Parses ONNX protobuf format directly
- Extracts graph structure, nodes, initializers
- Converts to internal `ModelGraph` representation

**Session-based Inference** (`include/tbn/runtime/model.hpp`):
- `TBNModel::Session` class for inference
- Sequential node execution
- Operator dispatch based on `op_type`

**Supported Operators**:
| Operator | Status |
|----------|--------|
| Conv | ✅ Optimized (im2col + GeMM) |
| Gemm | ✅ Optimized |
| MatMul | ✅ Optimized |
| Add | ✅ Broadcasting support |
| Relu | ✅ |
| Reshape | ✅ |
| Flatten | ✅ |
| MaxPool | ✅ |
| AveragePool | ✅ |
| GlobalMaxPool | ✅ |
| GlobalAveragePool | ✅ |

## Consequences

### Positive
- Standard model format (ONNX) with wide framework support
- Minimal export process changes required
- Full control over execution
- No ONNX Runtime dependency overhead
- Direct integration with optimized operators

### Negative
- Must implement operators manually
- No graph optimizations from ONNX Runtime
- Limited operator set compared to full ONNX Runtime

## Alternatives Considered

### 1. ONNX Runtime with Custom EP (Original Decision)
Use ONNX Runtime with a custom Execution Provider.
- Advantages: Full operator support, graph optimizations
- Disadvantages: Heavy dependency (~10MB), complex integration

### 2. TensorFlow Lite with Custom Ops
- Advantages: Mobile-focused
- Disadvantages: TensorFlow ecosystem lock-in

### 3. Apache TVM
- Advantages: Powerful compilation
- Disadvantages: Overkill for our use case

### 4. Custom ONNX Parser (Chosen)
Implement minimal ONNX parser.
- Advantages: Minimal overhead, full control, no dependency
- Disadvantages: Must implement operators

## Links

- ADR-0002: Quantization Strategy
- ADR-0003: Memory Layout
- ADR-0006: Conv2D Optimization
- ADR-0007: Pooling Operations
- `src/onnx_integration/onnx_parser.cpp` — parser implementation
- `include/tbn/runtime/model.hpp` — model and session