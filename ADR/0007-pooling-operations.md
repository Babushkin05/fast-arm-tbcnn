# ADR-0007: Pooling Operations

Date: 2026-04-24

## Status

Implemented

## Context

Pooling layers are essential components of CNNs, used for:
- Reducing spatial dimensions
- Providing translation invariance
- Reducing computation in subsequent layers

We need to support ONNX pooling operators to run standard CNN models.

## Decision

Implement pooling as standalone operations with a simple naive approach:

### Supported Operations
- **MaxPool2D**: Maximum value in each window
- **AvgPool2D**: Average value in each window
- **GlobalMaxPool2D**: Max over entire spatial dimensions
- **GlobalAvgPool2D**: Average over entire spatial dimensions

### Parameters
- `kernel_shape`: Size of pooling window
- `strides`: Step size (default: same as kernel)
- `pads`: Padding (ONNX format: [begin_h, begin_w, end_h, end_w])
- `ceil_mode`: Use ceil instead of floor for output size
- `count_include_pad`: Include padding in average calculation

### Implementation
Naive 6-nested loop implementation:
```
for (n, c, out_h, out_w, kernel_h, kernel_w)
    compute max/avg over window
```

## Consequences

### Positive
- Simple, correct implementation
- Supports all ONNX pooling attributes
- Works with any kernel size and stride
- Minimal memory overhead

### Negative
- Not optimized (pooling is typically not a bottleneck)
- Could benefit from SIMD for large kernels

## ONNX Operators Supported

| ONNX Operator | Status |
|---------------|--------|
| `MaxPool` | ✅ |
| `AveragePool` / `AvgPool` | ✅ |
| `GlobalMaxPool` | ✅ |
| `GlobalAveragePool` | ✅ |

## Alternatives Considered

### 1. SIMD-Optimized Pooling
Use NEON intrinsics for parallel max/avg computation.
- Advantages: Faster for large kernels
- Disadvantages: Pooling is rarely the bottleneck, not worth complexity

### 2. Separable Pooling
Decompose 2D pooling into two 1D passes.
- Advantages: Better cache behavior for large kernels
- Disadvantages: More complex, minimal benefit for typical 2×2, 3×3 kernels

## Implementation Details

**Files:**
- `include/tbn/operators/pooling.hpp` — interface
- `src/operators/pooling.cpp` — implementation
- `include/tbn/runtime/model.hpp` — ONNX integration

**Key Structures:**
```cpp
struct Pool2DParams {
    int64_t kernel_h, kernel_w;
    int64_t stride_h, stride_w;
    int64_t pad_h, pad_w;
    bool ceil_mode;
    bool count_include_pad;
};
```

## Performance Considerations

Pooling is typically not a performance bottleneck in CNNs:
- Most time spent in Conv2D layers
- Pooling windows are small (typically 2×2 or 3×3)
- Memory-bound operation, not compute-bound

Future optimization could include:
- SIMD for large kernels
- Integration with Conv2D for fused operations

## Links

- ADR-0006: Conv2D Optimization
- `tests/test_pooling.cpp` — tests
- ONNX Operator Specification: MaxPool, AveragePool
