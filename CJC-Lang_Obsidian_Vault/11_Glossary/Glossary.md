---
title: Glossary
tags: [glossary, reference]
status: Index
---

# Glossary

Short definitions for every recurring CJC-Lang term. Wikilinks point to the atomic note that explains the concept in depth.

## A

- **AD (Automatic Differentiation)** — forward-mode dual numbers + reverse-mode tape. See [[Autodiff]].
- **ADR (Architecture Decision Record)** — design decisions documented in `docs/adr/`.
- **Arena (frame arena)** — per-call-frame scratch memory that is reset on function return. See [[Frame Arena]].
- **AST** — abstract syntax tree produced by the parser; the input to the type checker and to [[cjc-eval]]. See [[AST]].

## B

- **Bastion** — doc-only spec for a 15-primitive statistical-computing library on top of CJC-Lang. See [[Bastion]].
- **Binned accumulator** — floating-point summation algorithm that groups terms by exponent bin for commutative, deterministic reductions. See [[Binned Accumulator]].
- **BLAS** — Basic Linear Algebra Subprograms. CJC-Lang ships its own and does **not** link external BLAS.
- **`BTreeMap` / `BTreeSet`** — ordered map/set. Required everywhere in CJC-Lang — `HashMap`/`HashSet` is banned for determinism. See [[Deterministic Ordering]].
- **Buffer** — the refcounted, copy-on-write byte storage under tensors. See [[COW Buffers]].
- **Builtin** — a native Rust function callable from CJC-Lang code. Must be registered in three places — see [[Wiring Pattern]].

## C

- **Capture analysis** — the HIR pass that identifies which outer variables a closure references. See [[Capture Analysis]].
- **CFG** — control-flow graph, built from MIR basic blocks. See [[CFG]].
- **`cjc-eval`** — the v1 AST tree-walking interpreter. See [[cjc-eval]].
- **`cjc-mir-exec`** — the v2 register-machine MIR executor. See [[cjc-mir-exec]].
- **CJC-Lang** — Computational Jacobian Core. The language this vault documents.
- **`cjcl`** — the CLI binary (was `cjc` before v0.1.4).
- **`.cjcl`** — source-file extension (was `.cjc` before v0.1.4).
- **Closure** — first-class function with captured environment. See [[Closures]].
- **COW (copy-on-write)** — shared buffers that clone only on mutation. See [[COW Buffers]].

## D

- **DCE (Dead Code Elimination)** — MIR optimizer pass. See [[MIR Optimizer]].
- **DataFrame** — tidyverse-style labeled table in `cjc-data`. See [[DataFrame DSL]].
- **Determinism contract** — the invariants CJC-Lang guarantees across runs and machines. See [[Determinism Contract]].
- **Dispatch layer** — the `cjc-dispatch` crate, which routes operator calls to typed kernels. See [[Dispatch Layer]].
- **DMRG** — density matrix renormalization group, a tensor-network quantum simulation method implemented in `cjc-quantum`.
- **Dominator tree** — a CFG analysis used by SSA construction and optimizer passes. See [[Dominator Tree]].
- **DSL** — domain-specific language. CJC-Lang uses DSL-style APIs for data (`cjc-data`) and plots (`cjc-vizor`).

## E

- **E-codes** — diagnostic error codes (E0001 through E8xxx). See [[Error Codes]].
- **Escape analysis** — MIR pass that classifies allocations as Stack / Arena / Rc. See [[Escape Analysis]].
- **Executor parity** — the invariant that AST-eval and MIR-exec produce bit-identical output for any valid program. See [[Parity Gates]].

## F

- **FMA (Fused Multiply-Add)** — a single-instruction `a*b + c` that changes rounding. **Banned** in CJC-Lang SIMD paths. See [[Float Reassociation Policy]].
- **Frame arena** — see Arena. [[Frame Arena]].

## G

- **GC / NoGC** — garbage collection is present in the value layer (`Rc`) but absent from proven "no-GC" paths validated by the [[NoGC Verifier]]. See [[Memory Model]].
- **Grammar of graphics** — the design philosophy behind [[Vizor]].

## H

- **Hardening** — the "beta hardening" phases that added audit tests, proptest, and fixtures. See [[Version History]].
- **HIR** — High-level Intermediate Representation; produced by `cjc-hir` from AST. See [[HIR]].

## I

- **`if` as expression** — planned language feature. See [[If as Expression]].

## K

- **Kahan summation** — compensated summation that reduces floating-point error. See [[Kahan Summation]].
- **KV cache** — paged transformer key/value cache (`paged_kv.rs` in `cjc-runtime`).

## L

- **Lexer** — tokenizer in `cjc-lexer`. See [[Lexer]].
- **libm** — the system math library. CJC-Lang **does not** use libm in hot paths — it ships its own. See [[Numerical Truth]].
- **Linalg** — the in-crate linear algebra (QR, LU, Cholesky, SVD, eigensolvers). See [[Linear Algebra]].
- **LLVM** — not used. Planned for Stage 3/4 as `cjc-codegen`. See [[Roadmap]].

## M

- **Match** — pattern-matching expression (structural destructuring over tuples and structs). See [[Patterns and Match]].
- **MIR** — Mid-level Intermediate Representation, register-based. See [[MIR]].
- **MPS** — matrix product state, one representation used by `cjc-quantum`.

## N

- **NoGC path** — a region of code proven not to allocate on the GC heap. See [[NoGC Verifier]].
- **NaN canonicalization** — in [[Binary Serialization]], all NaN bit patterns are rewritten to a single canonical NaN so two serialized snapshots hash identically.

## P

- **Parity gate** — a test that runs the same program through both executors and asserts byte-identical output. Gates G-8 and G-10 are the main ones. See [[Parity Gates]].
- **PINN** — physics-informed neural network; see [[PINN Support]].
- **Pratt parser** — operator-precedence parsing technique used by [[Parser]].

## Q

- **QAOA** — quantum approximate optimization algorithm, implemented in `cjc-quantum`.
- **QEC** — quantum error correction, surface code implementation in `cjc-quantum`.
- **Quantum simulation** — statevector + MPS + DMRG + density + stabilizer + VQE + QAOA + QML. See [[Quantum Simulation]].

## R

- **REINFORCE** — policy gradient algorithm used in [[Chess RL Demo]].
- **Register machine** — the execution model of [[cjc-mir-exec]].
- **`Rc`** — Rust refcounted pointer used for shared values. The only form of GC in CJC-Lang.

## S

- **SIMD** — single instruction, multiple data. CJC-Lang SIMD kernels avoid FMA and reassociation.
- **Snap** — the `cjc-snap` binary serialization format. See [[Binary Serialization]].
- **SplitMix64** — the only RNG in CJC-Lang. See [[SplitMix64]].
- **SSA** — static single-assignment form; used in MIR analysis. See [[SSA Form]].

## T

- **Tape** — reverse-mode AD tape in `cjc-ad`. See [[Autodiff]].
- **TCO** — tail call optimization, partially implemented. See [[Roadmap]] (S3-P1-07).
- **Tensor** — the core n-dimensional array type in `cjc-runtime`. See [[Tensor Runtime]].
- **`total_cmp`** — `f64::total_cmp` — the only permitted floating-point comparator for sorting in CJC-Lang. See [[Total-Cmp and NaN Ordering]].

## V

- **Value** — the runtime tagged-union type. See [[Value Model]].
- **Vizor** — grammar-of-graphics visualization library (`cjc-vizor`). See [[Vizor]].
- **VQE** — variational quantum eigensolver, implemented in `cjc-quantum`.

## W

- **Wiring pattern** — the rule that every new builtin must be registered in three places. See [[Wiring Pattern]].

## Related

- [[CJC-Lang Knowledge Map]]
- [[CJC-Lang Overview]]
