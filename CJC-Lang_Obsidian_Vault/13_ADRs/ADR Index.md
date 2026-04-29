---
title: ADR Index
tags: [adr, index, decisions]
status: Index
---

# ADR Index

Architecture Decision Records for CJC-Lang. Each ADR captures a single decision, why it was made, and what it constrains. The ADRs live in `docs/adr/` — these vault notes are one-screen summaries that link back to the source and connect decisions to concept notes.

**Numbering gap:** ADR-0006, ADR-0007, ADR-0008 do not exist in `docs/adr/`. The numbering jumps 0005 → 0009.

## Accepted decisions (live architecture)

| ID | Title | Status | Date | Summary |
|---|---|---|---|---|
| [[ADR-0001 Tree-form MIR]] | Tree-form MIR with separate derived CFG | Accepted | 2024-01-15 | Keep MIR as nested expressions; derive CFG as a view |
| [[ADR-0002 Kahan Accumulator]] | `KahanAccumulatorF64` for serial reductions | Accepted | 2024-01-15 | O(ε) error bound; order-dependent (serial only) |
| [[ADR-0003 Backward-compatible run_program]] | Stable executor entry points per capability | Accepted | 2024-01-20 | `run_program(program, seed)` signature never changes |
| [[ADR-0004 SplitMix64 RNG]] | SplitMix64 as the canonical CJC RNG | Accepted | 2024-01-15 | Seeded, cross-platform, zero-dep, ~1 ns/sample |
| [[ADR-0005 Binned Accumulator]] | Exponent-binned accumulator for order-invariant summation | Accepted | 2024-01-20 | Commutative; used by parallel reductions |
| [[ADR-0014 MIR Analysis Infrastructure]] | Loop analysis, reductions, verifier modules | Accepted | 2026-03-23 | Additive overlays on tree-form MIR |
| [[ADR-0015 PINN PDE Problem Suite]] | FD residuals, domain geometry, hard BCs | Accepted | 2026-04-11 | Burgers/Poisson/Heat solvers with 64 tests |
| [[ADR-0016 Language-Level GradGraph Primitives]] | `grad_graph_*` builtins via satellite dispatch | Accepted | 2026-04-26 | 24 new builtins; ambient thread-local graph; flips PINN to pure-CJC-Lang |
| [[ADR-0017 Adaptive TidyView Selection]] | Five-arm `AdaptiveSelection` enum, density-classified | Accepted | 2026-04-26 | Empty/All/SelectionVector/VerbatimMask + reserved Hybrid; sparse joins no longer pay dense costs |
| [[ADR-0018 Deterministic Adaptive Dictionary Engine]] | Byte-first categorical engine: `BytePool` + `AdaptiveCodes` + `BTreeMap` lookup | Accepted | 2026-04-28 | Phase 1 of TidyView v3; row-axis (ADR-0017) was adaptive, column-axis (categoricals) now adaptive too |

## Proposed decisions (not yet implemented)

| ID | Title | Status | Date | Summary |
|---|---|---|---|---|
| [[ADR-0009 Vec COW Array]] | `Rc<Vec<Value>>` for `Value::Array` and `Value::Tuple` | Proposed | 2025-01-01 | O(1) array passing; COW via `Rc::make_mut` |
| [[ADR-0010 Scope Stack SmallVec]] | SmallVec-backed scope frames | Proposed | 2025-01-01 | Deferred pending profile data |
| [[ADR-0011 Parallel Matmul]] | Rayon parallel matmul (feature-gated) | Proposed | 2025-01-01 | Uses [[ADR-0005 Binned Accumulator]] for determinism |
| [[ADR-0012 CFG Phi Nodes]] | True CFG with phi nodes and use-def chains | Proposed | 2025-01-01 | Extends [[ADR-0001 Tree-form MIR]] |
| [[ADR-0013 Package Manager]] | Minimal `cjc.toml` package manager | Proposed | 2026-03-22 | Git-based deps, no diamonds in v1 |

## Decision graph

```
ADR-0001 (Tree-form MIR) ──┬──> ADR-0003 (stable entry points)
                           └──> ADR-0012 (CFG phi nodes — *extends*, does not supersede)

ADR-0002 (Kahan, serial) ─────> ADR-0011 (parallel matmul — switches to Binned)
ADR-0005 (Binned, commutative) ┘

ADR-0004 (SplitMix64) ────────> every deterministic RNG user

ADR-0014 (MIR analysis) ──────> future optimizer passes (LICM, CSE, …)
```

## Reading order for new contributors

1. [[ADR-0001 Tree-form MIR]] — the MIR data model everything else builds on
2. [[ADR-0003 Backward-compatible run_program]] — how the executor API stays stable
3. [[ADR-0002 Kahan Accumulator]] + [[ADR-0005 Binned Accumulator]] — the two-accumulator determinism story
4. [[ADR-0004 SplitMix64 RNG]] — the other half of determinism
5. [[ADR-0014 MIR Analysis Infrastructure]] — what's live on top of MIR today
6. Proposed ADRs last — they are forward-looking

## Related

- [[Compiler Source Map]]
- [[Runtime Source Map]]
- [[Determinism Contract]]
- [[MIR]]
