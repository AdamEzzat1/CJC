# CJC Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the CJC compiler project.
ADRs document significant architectural decisions, the context that drove them, and their consequences.

## Index

| ADR | Title | Status | Crates Affected |
|-----|-------|--------|-----------------|
| [ADR-0001](ADR-0001-tree-form-mir-separate-cfg.md) | Tree-Form MIR + Separate CFG Analysis View | Accepted | cjc-mir, cjc-mir-exec |
| [ADR-0002](ADR-0002-kahan-accumulator-f64.md) | KahanAccumulatorF64 for Deterministic Serial Summation | Accepted | cjc-repro, cjc-runtime |
| [ADR-0003](ADR-0003-backward-compatible-run-program.md) | Backward-Compatible run_program() API Surface | Accepted | cjc-mir-exec |
| [ADR-0004](ADR-0004-splitmix64-rng.md) | SplitMix64 as the Canonical CJC RNG | Accepted | cjc-repro, cjc-runtime |
| [ADR-0005](ADR-0005-exponent-binned-accumulator.md) | Exponent-Binned Accumulator for Order-Invariant Summation | Accepted | cjc-runtime |
| [ADR-0009](ADR-0009-vec-cow-array-tuple.md) | Vec COW for Value::Array and Value::Tuple | Proposed | cjc-runtime, cjc-eval, cjc-mir-exec, cjc-types, cjc-hir, cjc-data |
| [ADR-0010](ADR-0010-scope-stack-smallvec.md) | Scope Stack SmallVec Optimization | Proposed | cjc-mir-exec |
| [ADR-0011](ADR-0011-parallel-matmul-rayon.md) | Parallel Matmul via Rayon (Optional Feature Gate) | Proposed | cjc-runtime |
| [ADR-0012](ADR-0012-true-cfg-phi-nodes.md) | True CFG with Phi Nodes and Use-Def Chains | Proposed | cjc-mir |

## Status Definitions

| Status | Meaning |
|--------|---------|
| **Accepted** | Decision implemented and in production. Reverting would require significant work. |
| **Proposed** | Decision documented and approved for implementation; work not yet started or in progress. |
| **Deferred** | Decision postponed pending profiling data or other prerequisites. |
| **Superseded** | Replaced by a newer ADR (referenced in the superseding document). |

## Decision Relationships

```
ADR-0001 (Tree-Form MIR)
  └── enables ADR-0012 (CFG already separate — phi nodes extend the existing CFG view)

ADR-0002 (Kahan Accumulator)
  └── enables ADR-0011 (BinnedAccumulator provides commutative parallel reduction;
       Kahan retains serial path)

ADR-0004 (SplitMix64 RNG)
  └── underpins ADR-0011 determinism tests (Tensor::randn used in matmul benchmarks)

ADR-0005 (Binned Accumulator)
  └── required by ADR-0011 (parallel matmul needs commutative accumulation)

ADR-0009 (Vec COW) — MUST land before ADR-0010
  └── scope stack profiling needs realistic Value::Array allocation data;
       COW changes the allocation pattern significantly

ADR-0003 (run_program API)
  └── guides ADR-0011 (parallel path exposed via feature flag, not a new entry point)
```

## ADR Template

To create a new ADR, copy this template:

```markdown
# ADR-NNNN: [Title]

**Status:** [Accepted | Proposed | Deferred | Superseded]
**Date:** [YYYY-MM-DD]
**Deciders:** [Role names]
**Supersedes:** [none | ADR-XXXX]

## Context
[Problem statement: what architectural pressure forced this decision?]

## Decision
[Exact decision taken, with API surface affected]

## Rationale
[Why this option over alternatives; trade-offs accepted]

## Consequences
[Positive outcomes; known limitations; what changes if this is reversed]

## Implementation Notes
- Crates affected: [list]
- Files to touch: [list with line ranges]
- Regression gate: `cargo test --workspace` must pass with 0 failures
```

## Numbering Convention

- ADR-0001 through ADR-0008: Core architecture decisions (made during Stage 1 and 2.0)
- ADR-0009 through ADR-0012: Stage 2 hardening decisions (proposed during Phase 2 audit)
- ADR-0013+: Stage 3 and beyond
