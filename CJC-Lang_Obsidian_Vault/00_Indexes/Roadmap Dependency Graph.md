---
title: Roadmap Dependency Graph
tags: [concept-graph, roadmap]
status: Graph hub
---

# Roadmap Dependency Graph

Which roadmap items block which. A plan for the order of future work.

## Reading guide

- **→** means "is a prerequisite for"
- IDs are from `docs/spec/stage3_roadmap.md`
- Concept links point to vault notes

## Dependency chains

### Infrastructure first (P0)

```
S3-P0-05 (runtime submodule split)
  → enables faster iteration on all runtime-touching tasks
  → prerequisite for S3-P2-01..04 (ML types need clean module boundaries)

S3-P0-06 (proptest)
  → independent; unblocks confidence in parser/type checker edits

S3-P0-07 (fixture runner)
  → prerequisite for reliable regression detection across all later work
```

### Compiler depth (P1)

```
S3-P1-06 (SSA + use-def + dom tree)
  → enables smarter DCE in [[MIR Optimizer]]
  → prerequisite for LLVM backend (S3-P3-01) because codegen wants SSA
  → helps S3-P1-07 (TCO extension) by giving precise liveness

S3-P1-07 (TCO for conditional branches + mutual recursion)
  → blocked by: nothing hard, but cleaner with S3-P1-06
  → requires: exec_body to propagate MirExecError::TailCall through branches

S3-P1-08 (Shape inference: E0500/E0501/E0502)
  → prerequisite for S3-P2-02/03/04 (ML types want shape-aware errors)
  → independent of compiler-depth chain otherwise
```

### Performance (P1)

```
S3-P1-01 (Vec COW for Array/Tuple)
  → prerequisite for predictable memory profile of [[ML Primitives]]
  → must not break [[Parity Gates]]

S3-P1-02 (Parallel matmul with rayon)
  → blocked by: proving [[Binned Accumulator]] commutativity across threads
  → independent of other roadmap items
  → qualifies the "zero external deps" claim — see [[Open Questions]]
```

### Language completeness (P1)

```
S3-P1-03 (extended numeric types: i8..u128, f16, Complex)
  → prerequisite for S3-P2-02 (quantized tensors need int8)

S3-P1-04 / S3-P1-05 (Set, Queue, Option, Result, Range, Slice as Type variants)
  → Option/Result must NOT break Value::Enum runtime representation
  → independent of compiler-depth chain
```

### ML types (P2)

```
S3-P2-01 (DType enum)
  → prerequisite for S3-P2-02 (QuantizedTensor dispatches on DType)
  → prerequisite for S3-P2-03 (MaskTensor needs DType to advertise as Bool)

S3-P2-04 (SparseTensor method dispatch)
  → independent — wires existing [[Sparse Linear Algebra]] into the value layer
```

### CLAUDE.md features (not all in Stage 3 ID space)

```
"if as expression"
  → independent; requires type unification of both branches
  → may benefit from S3-P1-08 (shape inference uses expression types)

Default parameters
  → prerequisite for variadic functions
  → simplifies many builtin signatures

Variadic functions
  → blocked by: default parameters (for sane call-site lowering)
  → required by: numerical solver stubs (ode_step with optional args)

Decorators
  → blocked by: closures (done)
  → independent of other roadmap items

Module system wiring
  → unblocked today (the crate exists, ~1,183 LOC) — see [[Module System]]
  → prerequisite for multi-file example programs
  → likely prerequisite for LSP (S3-P3-02)

MIR integration for autodiff
  → blocked by: nothing hard, but S3-P1-06 (SSA) helps
  → prerequisite for production-grade ML training

Sparse eigensolvers (Lanczos, Arnoldi)
  → blocked by: S3-P2-04 (sparse method dispatch)
  → prerequisite for large PCA / spectral methods in [[Bastion]]
```

### Stage 4 (P3)

```
S3-P3-01 (LLVM / Cranelift backend)
  → blocked by: S3-P1-06 (SSA), S3-P2-01 (DType)
  → highest-impact single item — unlocks 10-100× speedup
  → largest risk — touches compiler, runtime, and test infrastructure

S3-P3-02 (LSP)
  → blocked by: module system wiring
  → lower risk; builds on [[Language Server]]

WASM target (from CLAUDE.md, implied)
  → blocked by: module system, runtime dep audit
  → unlocks browser chess RL without the JS frontend

File I/O / JSON (Stage 4 preview)
  → independent; small self-contained additions
```

## Summary order (my recommendation)

Given dependencies, the **lowest-regret** order is:

1. S3-P0-05, 06, 07 — infrastructure
2. Module system wiring — low cost, high clarity win, documents itself
3. S3-P1-06 — SSA + dom tree — unlocks backend and optimizer
4. S3-P1-07 — TCO extension — builds on SSA
5. S3-P1-01 — Vec COW — localized perf win
6. S3-P2-01 — DType — foundation for ML types
7. `if` as expression — small user-facing polish with large syntactic leverage
8. S3-P1-02 — parallel matmul — once binned-accumulator correctness is proved
9. S3-P1-03, 04, 05 — numeric and collection types
10. S3-P1-08 — shape inference — once type system is richer
11. S3-P2-02, 03, 04 — ML types on top of DType
12. MIR-AD integration — prepare for Stage 4
13. Module system examples + LSP (S3-P3-02)
14. LLVM backend (S3-P3-01) — the big one
15. WASM target + File I/O + JSON — Stage 4 polish

## Related

- [[Roadmap]]
- [[Open Questions]]
- [[Documentation Gaps]]
- [[CJC-Lang Knowledge Map]]
