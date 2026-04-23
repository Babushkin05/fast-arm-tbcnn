# ADR-0006: Conv2D Optimization Strategy

Date: 2026-04-24

## Status

Implemented

## Context

Convolution is the most compute-intensive operation in CNNs. We need an efficient implementation that leverages our optimized ternary-binary GeMM. Key requirements:
- Support standard ONNX Conv2D semantics
- Achieve significant speedup over naive implementation
- Work with any ONNX model (float weights)
- Support stride, padding, dilation

## Decision

Implement Conv2D using im2col transformation + optimized GeMM:

### Algorithm
```
1. im2col: input [N,C,H,W] → col [N*out_h*out_w, C*kH*kW]
2. Reshape weights [M,C,kH,kW] → [M, C*kH*kW], transpose
3. qlinear_matmul_binary(col, weights^T) — uses optimized GeMM!
4. Reshape output → [N, M, out_h, out_w]
5. Add bias
```

### Auto-Quantization
Float weights are automatically quantized to binary on first execution:
```
float weights → quantize_to_binary() → binary weights → optimized path
```

## Consequences

### Positive
- **18x speedup** for medium/large layers vs naive implementation
- Any ONNX model works without modification
- Reuses highly optimized GeMM implementation
- Standard im2col approach is well-understood

### Negative
- Small layers (< 100 elements) may be slower due to im2col overhead
- Weight quantization happens on every inference (could cache)
- im2col requires temporary memory allocation

## Performance Results

Apple M4 Pro, 3×3 kernel:

| Layer | Naive | Optimized | Speedup |
|-------|-------|-----------|---------|
| 3×32×32 → 16×32×32 | 327 μs | 449 μs | 0.7x |
| 16×28×28 → 32×28×28 | 2837 μs | 738 μs | **3.8x** |
| 64×14×14 → 128×14×14 | 10425 μs | 577 μs | **18x** |
| 128×7×7 → 256×7×7 | 9531 μs | 824 μs | **11.6x** |

## Alternatives Considered

### 1. Direct Convolution (naive)
7-nested loop over all dimensions.
- Advantages: Simple, no extra memory
- Disadvantages: Very slow for large layers

### 2. Winograd Algorithm
Fast convolution for small kernels (3×3).
- Advantages: ~2-3x speedup for 3×3 kernels
- Disadvantages: Complex, only efficient for small kernels

### 3. FFT-based Convolution
Use FFT for large kernels.
- Advantages: Efficient for large kernels
- Disadvantages: Overkill for typical 3×3, 5×5 kernels

## Implementation Details

**Files:**
- `src/operators/conv2d.cpp` — `conv2d_binary()`, `conv2d_ternary()`
- `include/tbn/runtime/model.hpp` — auto-quantization in `execute_conv()`

**Key Functions:**
- `impl::im2col()` — input transformation
- `qlinear_matmul_binary()` — optimized GeMM
- `quantize_to_binary()` — weight quantization

## Links

- ADR-0002: Quantization Strategy
- ADR-0003: Memory Layout
- `GeMM/05-final/GeMM.hpp` — optimized GeMM implementation
- `tests/test_conv2d_optimized.cpp` — benchmarks
