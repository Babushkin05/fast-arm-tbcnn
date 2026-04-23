# ADR-0003: Memory Layout for Ternary-Binary Weights

Date: 2026-04-21

## Status

Implemented (2026-04-24)

## Context

Ternary-binary neural networks require efficient storage of weights that can take only three values: -1, 0, +1. Standard float32 storage wastes memory and cache bandwidth. We need a memory layout that:
- Minimizes memory footprint
- Enables efficient SIMD operations
- Works well with ARM NEON instructions
- Supports per-channel quantization scales

## Decision

Implemented packed bit representation with optimized GeMM access patterns:

### Current Implementation

**Binary Weights** (`DataType::BINARY`):
- `BINARY_ZERO = 0` → represents -1
- `BINARY_ONE = 1` → represents +1
- Stored as `uint8_t` (BinaryWeight type)
- Used in `qlinear_matmul_binary()`

**Ternary Weights** (`DataType::TERNARY`):
- Values: -1, 0, +1 as `int8_t`
- Stored as `int8_t` (TernaryWeight type)
- Used in `qlinear_matmul_ternary()`

**Packed Weights** (for optimized GeMM):
- `BinaryPackedWeights`: 8 values/byte bit-packing
- `TernaryPackedWeights`: 4 values/byte (2 bits each)
- Defined in `include/tbn/memory/packed_weights.hpp`

### Blocked GeMM Layout

From `GeMM/05-final/GeMM.hpp`:
- Tiling parameters: `mblk=64, nblk=64, kblk=128`
- Cache-aligned memory access
- L1/L2 cache optimization

## Consequences

### Positive
- **16x memory reduction** for binary weights vs float32
- Efficient NEON SIMD operations
- Cache-friendly blocked access patterns
- **23x speedup** for GeMM operations

### Negative
- Unpacking overhead for non-SIMD operations
- Alignment requirements for optimal performance
- Complexity in weight management

## Performance Results

Memory reduction combined with SIMD optimization:
| Matrix Size | Memory Savings | Speedup |
|-------------|---------------|---------|
| 128×128 | 16x | 4.6x |
| 512×512 | 16x | 16.3x |
| 1024×1024 | 16x | 22.9x |

## Implementation Details

**Files:**
- `include/tbn/runtime/types.hpp` — BINARY, TERNARY types
- `include/tbn/memory/packed_weights.hpp` — packed weight classes
- `GeMM/05-final/GeMM.hpp` — blocked GeMM with tiling

**Key Classes:**
```cpp
class BinaryPackedWeights {
    // 8 values per byte (1 bit each)
    BinaryWeight get_weight(int64_t index);
    void set_weight(int64_t index, BinaryWeight value);
};

class TernaryPackedWeights {
    // 4 values per byte (2 bits each)
    TernaryWeight get_weight(int64_t index);
    void set_weight(int64_t index, TernaryWeight value);
};
```

## Links

- ADR-0001: ONNX Integration
- ADR-0002: Quantization Strategy
- ADR-0006: Conv2D Optimization
- `include/tbn/memory/packed_weights.hpp` — packed weight classes
- `GeMM/05-final/GeMM.hpp` — blocked GeMM implementation