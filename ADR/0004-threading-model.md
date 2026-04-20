# ADR-0004: Threading Model for Inference

Date: 2026-04-21

## Status

Proposed

## Context

We need to decide how to utilize multiple CPU cores for ternary-binary neural network inference. The threading model must balance:
- Throughput for batch processing
- Latency for single inference
- Energy efficiency on mobile devices
- Cache coherency with our blocking strategy

## Decision

Use a hybrid threading model with two levels:

1. **Inter-layer parallelism**: Different layers can execute in parallel when dependencies allow
2. **Intra-layer parallelism**: Individual operations (Conv2D, Gemm) use multiple threads via OpenMP

The model uses a thread pool with these characteristics:
- Worker threads pinned to big cores on heterogeneous ARM processors
- Work-stealing queue for load balancing
- NUMA-aware memory allocation for multi-cluster systems
- Dynamic thread count based on workload and thermal state

## Consequences

### Positive
- Better utilization of multi-core ARM processors
- Reduced latency for multi-image inference
- Energy proportional computing (use fewer cores for light loads)
- Natural fit with ONNX Runtime's threading model

### Negative
- Increased complexity in synchronization
- Higher memory usage per thread
- Potential cache thrashing with multiple thread teams
- More complex debugging and profiling

## Alternatives Considered

### 1. Single-threaded with SIMD
Use only SIMD parallelism within operations.
Advantages: Simple implementation, predictable performance
Disadvantages: Poor multi-core utilization, high latency for batches

### 2. Layer-wise Pipeline
Each layer runs in its own thread with queues between layers.
Advantages: Good throughput, natural pipelining
Disadvantages: Memory overhead from intermediate buffers, complex scheduling

### 3. Single-thread per inference
Each inference request gets one thread.
Advantages: Simple resource management
Disadvantages: Poor cache locality, cannot optimize across layers

## Decision Drivers
1. Performance: Maximize throughput while maintaining low latency
2. Efficiency: Scale with workload and respect thermal constraints
3. Portability: Work across different ARM core configurations
4. Compatibility: Integrate with ONNX Runtime threading

## Links
- ADR-0001: ONNX Runtime Integration
- ADR-0003: Memory Layout
- Implementation: src/runtime/thread_pool.hpp
- Performance analysis: benchmarks/threading.md

## Future Considerations
- GPU offload for large batches
- Work-stealing between inferences
- Cache-aware thread placement
- Integration with Android's thread scheduling

## Testing Strategy
Measure:
- Throughput scaling with batch size
- Latency for single images
- Energy consumption per inference
- Performance on different core configurations (big.LITTLE, homogeneous)

## Decision

Proposed - The hybrid model provides flexibility to optimize for different scenarios while maintaining compatibility with existing infrastructure. Implementation will start with basic OpenMP parallelization and evolve to custom thread pool based on measured bottlenecks. Performance targets: 80% core utilization for batches, <10% latency increase for singles. Further optimization based on profiling results. Design ready for implementation phase. Final decision after prototype evaluation.