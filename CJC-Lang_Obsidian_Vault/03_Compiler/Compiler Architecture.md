---
title: Compiler Architecture
tags: [compiler, hub]
status: Implemented
---

# Compiler Architecture

CJC-Lang's compiler is a multi-stage pipeline that ends at **two** execution backends, not one. Both backends must agree byte-for-byte — that is the compiler's top-level correctness invariant, enforced by [[Parity Gates]].

## The pipeline

```
Source.cjcl
    │
    ▼
 [Lexer]            crates/cjc-lexer   — tokenize
    │  tokens
    ▼
 [Parser]           crates/cjc-parser  — Pratt parser
    │  AST
    ▼
 [Type Checker]     crates/cjc-types   — HM unification
    │  typed AST
    ▼
 ┌─────────────────────────┬─────────────────────────┐
 │                         │                         │
 ▼                         ▼                         │
[cjc-eval]            [HIR Lowering]                 │
tree-walk v1      crates/cjc-hir                     │
                    │  HIR                           │
                    ▼                                │
                  [MIR Lowering]                     │
                  crates/cjc-mir                     │
                    │  MIR                           │
                    ▼                                │
                  [MIR Optimizer]                    │
                  CF, DCE, CSE, SR, LICM, SCCP       │
                    │                                │
                    ▼                                │
                  [cjc-mir-exec]                     │
                  register machine v2                │
                                                     │
                                                     │
              ◄────────── Parity Gates ──────────────┘
```

## The crates by role

| Crate | Role | Stage |
|---|---|---|
| [[cjc-lexer]] | Tokenization | 1 |
| [[cjc-parser]] | Pratt parser → AST | 2 |
| [[cjc-ast]] | AST node definitions | Leaf data |
| [[cjc-types]] | Type system + HM inference | 3 |
| [[cjc-diag]] | Diagnostics, spans, error codes | cross-cutting |
| [[cjc-hir]] | AST → HIR lowering + [[Capture Analysis]] | 4 |
| [[cjc-mir]] | HIR → MIR lowering, CFG, SSA, optimizer, verifiers | 5 |
| [[cjc-eval]] | Tree-walking interpreter (v1) | leaf A |
| [[cjc-mir-exec]] | Register-machine executor (v2) | leaf B |
| [[cjc-dispatch]] | Multi-dispatch / operator resolution | cross-cutting |

## Key invariants

1. **Parity**: every program must produce identical output in [[cjc-eval]] and [[cjc-mir-exec]]. See [[Parity Gates]].
2. **Determinism**: all internal collections are `BTreeMap` / `BTreeSet`. See [[Deterministic Ordering]].
3. **NoGC zones**: functions marked `@nogc` must be statically proven allocation-free by the [[NoGC Verifier]].
4. **Reduction contract**: the optimizer may not reorder floating-point reductions that carry a reduction annotation (see `crates/cjc-mir/src/reduction.rs`). See [[Float Reassociation Policy]].
5. **Legality verifier**: `crates/cjc-mir/src/verify.rs` enforces CFG structural rules, SSA soundness, and the reduction contract before execution.

## Stages in detail

Each stage has its own note:

- [[Lexer]]
- [[Parser]]
- [[AST]]
- [[Type Checker]]
- [[HIR]]
- [[MIR]]
- [[MIR Optimizer]]
- [[SSA Form]]
- [[CFG]]
- [[Dominator Tree]]
- [[Loop Analysis]]
- [[NoGC Verifier]]
- [[Escape Analysis]]
- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[Dispatch Layer]]
- [[Parity Gates]]
- [[Diagnostics]]
- [[Error Codes]]

## What is NOT here

- **LLVM / native backend** — [[Roadmap]].
- **JIT** — not planned in near term.
- **WASM target** — [[Roadmap]].
- **Incremental compilation** — not implemented.

## Related

- [[CJC-Lang Overview]]
- [[Runtime Architecture]]
- [[Compiler Concept Graph]]
- [[Compiler Source Map]]
