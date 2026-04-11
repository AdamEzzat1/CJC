---
title: CJC-Lang Overview
tags: [hub, overview]
status: Implemented (for the claims it makes)
---

# CJC-Lang Overview

**CJC-Lang** (Computational Jacobian Core) is a deterministic numerical programming language implemented in Rust. It is under active development, currently at **v0.1.4**, with 21 workspace crates totalling ~96K lines of code and zero external runtime dependencies.

The language targets three workloads:
1. **Reproducible numerical computing** — same seed produces bit-identical output
2. **Machine learning pipelines** — tensors, autodiff, training loops
3. **Data analysis** — tidyverse-style DataFrame DSL

## One-Minute Summary

- **Two execution backends** that must agree bit-for-bit: a tree-walking AST interpreter ([[cjc-eval]]) and a MIR register-machine executor ([[cjc-mir-exec]]). This equivalence is enforced by [[Parity Gates]].
- **Zero external dependencies.** No LLVM, no libc math, no third-party crates in the hot path. Everything from lexer to linear algebra to quantum simulation is written in Rust inside the workspace.
- **Determinism is a hard constraint**, not a feature. See [[Determinism Contract]] — `BTreeMap` everywhere, [[SplitMix64]] RNG, [[Kahan Summation]], no FMA in SIMD kernels, `total_cmp` for float ordering.
- **Compiler pipeline**: [[Lexer]] → [[Parser]] → [[AST]] → [[Type Checker]] → [[HIR]] → [[MIR]] → [[MIR Optimizer]] → [[cjc-mir-exec]]. The AST branch feeds [[cjc-eval]] directly. See [[Compiler Architecture]].
- **Runtime surface**: ~334 builtin functions spanning math, statistics, distributions, hypothesis tests, linear algebra, FFT, signal processing, ML primitives, and data wrangling. See [[Runtime Architecture]] and [[Builtins Catalog]].

## Entry Points

| Topic | Start here |
|---|---|
| The map of the whole thing | [[CJC-Lang Knowledge Map]] |
| What makes it different | [[What Makes CJC-Lang Distinct]] |
| What it can actually do today | [[Current State of CJC-Lang]] — [[What CJC-Lang Can Already Do]] |
| The compiler internals | [[Compiler Architecture]] — [[Compiler Concept Graph]] |
| The runtime internals | [[Runtime Architecture]] — [[Runtime Concept Graph]] |
| The determinism promise | [[Determinism Contract]] — [[Numerical Truth]] |
| ML, tensors, AD | [[Tensor and Scientific Computing]] — [[Autodiff]] |
| Data, DataFrames, Vizor | [[Data Systems and CLI]] |
| Quantum, solvers | [[Advanced Computing in CJC-Lang]] |
| Roadmap and open issues | [[Roadmap]] — [[Open Questions]] |
| Terminology | [[Glossary]] |

## Version and Evolution

- **v0.1.0** — First MVP: lexer, parser, eval, types, dispatch, runtime, AD, data DSL.
- **Stage 2.0–2.4** — HIR, MIR, MIR-exec, closures with capture analysis, match + destructuring, for-loops, NoGC verifier, CF + DCE optimizer, parity gates.
- **v0.1.2** — Data science foundation, ML infrastructure, 30+ CLI commands.
- **v0.1.3** — `cargo install cjc` fixed.
- **v0.1.4** — Rebranded `cjc` → `cjcl` CLI and `.cjc` → `.cjcl` file extension. Crate: `cjc-lang`. (See [[Version History]].)

## Current Headline Numbers

These numbers come from `README.md` and `CLAUDE.md` as of 2026-04-09. They should be re-verified whenever this note is updated.

- **Crates**: 21 in `crates/` (one workspace)
- **LOC**: ~96K
- **Tests**: 3,700+ to 5,320+ workspace tests depending on configuration. (The README says 3,700+; CLAUDE.md notes 5,320 as of 2026-03-21. **Needs verification** of the exact current count.)
- **Builtins**: ~221+ per README, ~334 per code survey. (The higher number includes lower-level helpers; the README figure counts user-facing functions. **Needs verification**.)
- **Dependencies**: zero external runtime dependencies

## The Shape of the Project

```
Source.cjcl ─► Lexer ─► Parser ─► AST ─► TypeChecker ─► Typed AST
                                                            │
                             ┌──────────────────────────────┼──────────────────────────────┐
                             ▼                                                              ▼
                        cjc-eval                                                      HIR Lowering
                     (tree-walk v1)                                                          │
                                                                                              ▼
                                                                                      MIR Lowering
                                                                                              │
                                                                                              ▼
                                                                                       MIR Optimizer
                                                                                      (CF, DCE, CSE,
                                                                                       LICM, SCCP, ...)
                                                                                              │
                                                                                              ▼
                                                                                       cjc-mir-exec
                                                                                     (register VM v2)
```

Both leaves must produce byte-for-byte identical output for any program. That invariant is called the **parity contract**. See [[Parity Gates]].

## Related Notes

- [[What Makes CJC-Lang Distinct]]
- [[Current State of CJC-Lang]]
- [[Compiler Architecture]]
- [[Runtime Architecture]]
- [[Determinism Contract]]
- [[Roadmap]]
