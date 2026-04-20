# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADR) for the Ternary-Binary GeMM project. ADRs document important architectural decisions, their context, consequences, and alternatives considered.

## What is ADR?

An Architecture Decision Record is a document that captures a significant architectural decision made along with its context and consequences. ADRs are immutable - if a decision changes, a new ADR is created that supersedes the previous one.

## ADR Format

Each ADR follows the template in `0000-template.md`:
- **Title**: Brief description of the decision
- **Status**: Proposed, Accepted, Rejected, or Superseded
- **Context**: The problem and constraints
- **Decision**: What was decided
- **Consequences**: Positive and negative impacts
- **Alternatives**: Other options considered
- **Links**: References to related ADRs and code

## Current ADRs

| Number | Title | Status | Date |
|--------|-------|--------|------|
| ADR-0001 | ONNX Runtime Integration | Proposed | 2026-04-21 |
| ADR-0002 | Quantization Strategy | Proposed | 2026-04-21 |
| ADR-0003 | Memory Layout for Ternary-Binary Weights | Proposed | 2026-04-21 |
| ADR-0004 | Threading Model for Inference | Proposed | 2026-04-21 |
| ADR-0005 | Error Handling Strategy | Proposed | 2026-04-21 |

## Proposed ADRs

- ADR-0006: API Design for TBN Runtime
- ADR-0007: Model Format and Serialization
- ADR-0008: Testing Strategy for Quantized Models
- ADR-0009: Performance Profiling and Benchmarking
- ADR-0010: Mobile Platform Integration

## Creating New ADRs

1. Copy `0000-template.md` to `NNNN-title-with-dashes.md`
2. Fill in all sections following the template
3. Update this README to include the new ADR
4. Link to/from related ADRs

## ADR Lifecycle

1. **Proposed**: Initial draft, open for discussion
2. **Accepted**: Approved after review, ready for implementation
3. **Rejected**: Decision not to implement
4. **Superseded**: Replaced by newer ADR

## Decision Process

1. Create ADR in Proposed status
2. Discuss in team/PR comments
3. Update based on feedback
4. Accept or Reject after consensus
5. Implement if Accepted

## References

- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
- [Martin Fowler on ADRs](https://martinfowler.com/articles/agile-architecture.html#ArchitectureDecisions)