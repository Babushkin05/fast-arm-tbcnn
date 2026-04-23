# ADR-0002: Quantization Strategy for TBN

Date: 2026-04-21

## Status

Implemented (2026-04-24)

## Context

We need to determine how and when to convert FP32 model weights to ternary values (-1, 0, +1) for efficient execution on ARM processors. Key questions include:
- When to perform quantization: offline or runtime?
- What granularity level: per-tensor or per-channel?
- How to handle activations: keep FP32 or also quantize?
- What quantization threshold to use?

## Decision

Implemented runtime quantization with on-the-fly conversion:

### Weight Quantization
- **Float → Binary**: `quantize_to_binary(tensor, threshold=0.0f)`
- **Float → Ternary**: `quantize_to_ternary(tensor, threshold_low, threshold_high)`
- Automatic quantization in `execute_conv()` when float weights detected

### Activation Quantization
- On-the-fly quantization to ternary during GeMM computation
- Thresholds: `threshold_low=-0.1f`, `threshold_high=0.1f`
- No permanent storage of quantized activations

### Binary Encoding
- `BINARY_ZERO = 0` → represents -1
- `BINARY_ONE = 1` → represents +1
- Stored as `uint8_t` (8 values per byte conceptually, 1 byte used)

## Consequences

### Positive
- Any ONNX model works without modification
- No preprocessing step required
- Runtime flexibility
- **23x speedup** for GeMM with binary weights

### Negative
- Quantization overhead on first inference
- Could benefit from caching quantized weights
- Accuracy loss from weight quantization (acceptable for inference)

## Performance Results

### GeMM Speedup (Float × Binary)
| Size | Float × Binary | Regular GEMM | Speedup |
|------|---------------|--------------|---------|
| 128×128 | 880 μs | 4.1 ms | **4.6x** |
| 256×256 | 3.9 ms | 31.0 ms | **8.0x** |
| 512×512 | 22.8 ms | 371.6 ms | **16.3x** |
| 1024×1024 | 133 ms | 3045 ms | **22.9x** |

### Conv2D Speedup (with im2col + GeMM)
| Layer | Naive | Optimized | Speedup |
|-------|-------|-----------|---------|
| 64×14×14 → 128×14×14 | 10425 μs | 577 μs | **18x** |

## Implementation Details

**Files:**
- `src/operators/quantized_gemm_neon.cpp` — `qlinear_matmul_binary()`, `quantize_to_binary()`
- `src/operators/quantized_gemm.cpp` — fallback implementations
- `include/tbn/operators/quantized_gemm.hpp` — API

**Key Functions:**
```cpp
Tensor quantize_to_binary(const Tensor& weights, float threshold = 0.0f);
Tensor quantize_to_ternary(const Tensor& weights, float low = -0.1f, float high = 0.1f);
Tensor qlinear_matmul_binary(const Tensor& a, const Tensor& b_binary, float scale);
```

## Alternatives Considered

### 1. Pure Offline Quantization
Quantize in advance and save as custom format.
- Advantages: Fast loading
- Disadvantages: Requires converter tool

### 2. Training-Aware Quantization
Train with ternary constraints.
- Advantages: Best accuracy
- Disadvantages: Requires retraining

## Links

- ADR-0001: ONNX Integration
- ADR-0003: Memory Layout
- ADR-0006: Conv2D Optimization
- `src/operators/quantized_gemm_neon.cpp` — implementation
- `GeMM/05-final/GeMM.hpp` — blocked GeMM algorithm