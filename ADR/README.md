# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADR) for the Ternary-Binary Neural Network (TBN) project. ADRs document important architectural decisions, their context, consequences, and alternatives considered.

## What is ADR?

An Architecture Decision Record is a document that captures a significant architectural decision made along with its context and consequences. ADRs are immutable - if a decision changes, a new ADR is created that supersedes the previous one.

## ADR Format

Each ADR follows the template in `0000-template.md`:
- **Title**: Brief description of the decision
- **Status**: Proposed, Accepted, Implemented, Rejected, or Superseded
- **Context**: The problem and constraints
- **Decision**: What was decided
- **Consequences**: Positive and negative impacts
- **Alternatives**: Other options considered
- **Links**: References to related ADRs and code

## Current ADRs

| Number | Title | Status | Implementation |
|--------|-------|--------|----------------|
| ADR-0001 | ONNX Runtime Integration | ✅ Implemented | `onnx_parser.cpp`, `model.hpp` |
| ADR-0002 | Quantization Strategy | ✅ Implemented | `quantized_gemm.cpp`, `quantize_to_binary()` |
| ADR-0003 | Memory Layout for Ternary-Binary Weights | ✅ Implemented | `packed_weights.hpp`, GeMM blocking |
| ADR-0004 | Threading Model for Inference | ⏳ Proposed | — |
| ADR-0005 | Error Handling Strategy | ✅ Implemented | `errors.hpp`, custom exceptions |
| ADR-0006 | Conv2D Optimization Strategy | ✅ Implemented | `conv2d.cpp`, im2col + GeMM |
| ADR-0007 | Pooling Operations | ✅ Implemented | `pooling.cpp` |

## Implementation Status Summary

### Core Components (Implemented)

| Component | Status | Performance |
|-----------|--------|-------------|
| ONNX Parser | ✅ Complete | Models load correctly |
| Optimized GeMM | ✅ Complete | 23x speedup vs float |
| Conv2D (im2col + GeMM) | ✅ Complete | 18x speedup vs naive |
| Auto-quantization | ✅ Complete | Any ONNX → optimized path |
| Pooling | ✅ Complete | MaxPool, AvgPool, Global |
| Tensor infrastructure | ✅ Complete | NCHW layout, quantized types |
| Error handling | ✅ Complete | Custom exceptions, logging |

### Pending Components

| Component | Status | Notes |
|-----------|--------|-------|
| Threading | ⏳ Proposed | Multi-core parallelism |
| BatchNorm folding | ⏳ Planned | Fold into conv weights |
| Mobile deployment | ⏳ Planned | Android, iOS, RPi |

## Creating New ADRs

1. Copy `0000-template.md` to `NNNN-title-with-dashes.md`
2. Fill in all sections following the template
3. Update this README to include the new ADR
4. Link to/from related ADRs

## ADR Lifecycle

1. **Proposed**: Initial draft, open for discussion
2. **Accepted**: Approved after review, ready for implementation
3. **Implemented**: Code complete and tested
4. **Rejected**: Decision not to implement
5. **Superseded**: Replaced by newer ADR

## Decision Process

1. Create ADR in Proposed status
2. Discuss in team/PR comments
3. Update based on feedback
4. Accept or Reject after consensus
5. Implement if Accepted
6. Mark as Implemented when complete

## References

- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
- [Martin Fowler on ADRs](https://martinfowler.com/articles/agile-architecture.html#ArchitectureDecisions)
