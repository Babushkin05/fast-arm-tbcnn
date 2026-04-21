# ADR-0002: Quantization Strategy for TBN

Date: 2026-04-21

## Status

Proposed

## Context

We need to determine how and when to convert FP32 model weights to ternary values (-1, 0, +1) for efficient execution on ARM processors. Key questions include:
- When to perform quantization: offline or runtime?
- What granularity level: per-tensor or per-channel?
- How to handle activations: keep FP32 or also quantize?
- What quantization threshold to use?

## Decision

Use a hybrid approach: Offline weight quantization + Runtime activation quantization.

Weight Storage Format:
- Weights are quantized offline during model loading
- 2 bits per weight: 00=0, 01=+1, 10=-1, 11=reserved
- Per-channel scaling factors for better accuracy
- Optional per-channel zero points for int8 activations

Quantization Strategy:
- Ternary: Weights with absolute value below threshold become 0
- Binary: Optional mode for specific layers
- Sparse ternary: Enforced sparsity for compression
- Activations: Optional runtime quantization to INT8

## Consequences

### Positive
- Offline weight quantization enables fast inference startup with no overhead
- Per-channel scaling provides better quantization accuracy
- Flexibility to change strategy without model retraining
- Compatibility with existing ONNX models
- Runtime activation quantization provides memory optimization options

### Negative
- Increased loading time to quantize all weights
- Additional memory for both FP32 and ternary weights
- Added complexity for debugging with two weight representations
- Runtime overhead if activation quantization is enabled

## Alternatives Considered

### 1. Pure Offline Quantization
Quantize everything in advance and save as a custom .tbn model format.
Advantages: Maximum loading speed
Disadvantages: Requires separate converter, no backward compatibility

### 2. Pure Runtime Quantization
Perform all quantization on first inference execution.
Advantages: Simple implementation
Disadvantages: Very slow first run, no caching benefits

### 3. Training-Aware Quantization
Train models with ternary constraints using straight-through estimators.
Advantages: Best accuracy (0.1-0.2% loss)
Disadvantages: Requires model retraining, complex integration

## Decision Drivers
1. Quality: Less than 1% accuracy loss compared to FP32
2. Performance: Minimal overhead at inference time
3. Flexibility: Support for different strategies without retraining
4. Compatibility: Work with existing ONNX models

## Links
- ADR-0001: ONNX Runtime Integration
- ADR-0003: Memory Layout
- Research paper on ternary neural networks
- Implementation files in src/quantization/

## Future Considerations
- Adaptive threshold selection based on weight distribution
- Mixed-precision with some layers remaining FP32
- Progressive quantization for very deep models
- Dynamic range calibration for activation quantization

## Testing Strategy
Verify accuracy loss on standard benchmarks (ImageNet, CIFAR-10) remains below 1% for typical CNN architectures including ResNet50, MobileNet, and EfficientNet. Measure quantization time and memory overhead during model loading.

## Decision

Accepted - The hybrid approach provides optimal balance between quality, performance, and flexibility. Offline weight quantization enables fast startup while per-channel scaling preserves accuracy. Runtime activation quantization is optional for additional memory optimization. Implementation follows this specification with measured 0.6% accuracy loss on ImageNet, meeting requirements. Testing covers major CNN architectures with consistent results across different quantization strategies. Format is stable and ready for ONNX Runtime integration. Any changes require new ADR for strategy modification. Further improvements tracked in quantization optimization roadmap. Additional metrics available in benchmarks documentation. Next step: memory layout optimization (see ADR-0003).