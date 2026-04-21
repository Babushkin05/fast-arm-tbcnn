# ADR-0001: ONNX Runtime Integration

Date: 2026-04-21

## Status

Proposed

## Context

For the ternary-binary neural network, we need a way to load and execute models from standard frameworks (PyTorch, TensorFlow). The solution must:
- Support standard model formats
- Enable replacement of standard operations with custom ternary implementations
- Require minimal changes to the model export process
- Provide good performance on ARM processors

## Decision

Use ONNX Runtime with a custom Execution Provider (EP) for ternary-binary operations.

The architecture consists of three layers:
1. ONNX Runtime Core - handles standard operations and graph execution
2. TBN Execution Provider - implements custom ternary operations
3. Quantization Layer - converts FP32 weights to ternary representation

This approach allows us to leverage ONNX Runtime's mature infrastructure while replacing only the compute-intensive operations (Conv2D, Gemm) with our optimized ternary implementations.

## Consequences

### Positive
- Standard model format (ONNX) with wide framework support
- Minimal export process changes required
- Built-in support for all standard operations
- Good ARM performance through ONNX Runtime optimizations
- Access to ONNX ecosystem tools
- Ability to use existing graph optimizations

### Negative
- Additional dependency (ONNX Runtime ~10MB)
- Need to understand ONNX Runtime APIs and patterns
- Overhead from ONNX Runtime abstractions
- Must follow ONNX Runtime extension patterns

## Alternatives Considered

### 1. Custom ONNX Parser
Implementation of a complete ONNX parser with ternary operation support.
Advantages: Full control, minimal overhead
Disadvantages: Must implement 100+ operations, complex maintenance

### 2. TensorFlow Lite with Custom Ops
Use TFLite infrastructure with custom operations for ternary compute.
Advantages: Mobile-focused, quantization-aware training support
Disadvantages: Tight coupling to TensorFlow ecosystem, less flexibility

### 3. Apache TVM
Use TVM's compilation infrastructure for automatic optimization.
Advantages: Powerful compilation system, automatic optimizations
Disadvantages: Complex integration, overkill for our specific use case

## Decision Drivers
1. Time-to-market: Fast integration with existing models
2. Performance: Effective use of ARM NEON instructions
3. Maintainability: Standard tools and formats
4. Flexibility: Ability to customize key operations

## Links
- ADR-0002: Quantization Strategy
- ADR-0003: Memory Layout
- ONNX Runtime Custom EP Documentation
- Related implementation files in src/runtime/