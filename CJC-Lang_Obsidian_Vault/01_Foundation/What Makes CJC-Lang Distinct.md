---
title: What Makes CJC-Lang Distinct
tags: [foundation, identity]
status: Implemented (for the claims it makes)
---

# What Makes CJC-Lang Distinct

CJC-Lang is not trying to be the fastest, the most ergonomic, or the most general-purpose language. It is trying to be a language where **the same program with the same inputs produces bit-identical output everywhere**, while also covering enough numerical ground to run real ML and scientific workloads.

The distinctive combination is:

## 1. Determinism as a hard constraint

Most languages treat determinism as something you layer on top with care. CJC-Lang treats it as an **invariant the compiler and runtime refuse to violate**:

- All internal maps use `BTreeMap` / `BTreeSet`. `HashMap` is essentially banned in the hot path (see [[Deterministic Ordering]]).
- Floating-point reductions go through [[Kahan Summation]] or [[Binned Accumulator]] — never a naive `+` loop.
- The RNG is [[SplitMix64]] with explicit seed threading; nothing reads from `/dev/urandom` implicitly.
- SIMD kernels avoid FMA to keep bit-identical results across architectures.
- Sorting uses `f64::total_cmp` so NaN ordering is stable.
- The [[NoGC Verifier]] proves statically that marked functions never trigger allocation.

See [[Determinism Contract]] for the full list.

## 2. Two executors that must agree

CJC-Lang has *two* execution backends:
- **[[cjc-eval]]** — a tree-walking AST interpreter (v1).
- **[[cjc-mir-exec]]** — a register-machine executor over [[MIR]] (v2).

Every program must produce byte-identical output in both. This is enforced by [[Parity Gates]], a dedicated test suite (G-8, G-10, and related gates). The parity requirement is what forces shared dispatch logic in [[cjc-dispatch]] and [[cjc-runtime]]'s `builtins.rs` — both executors call into the same stateless layer.

This is unusual. Most languages have one reference implementation and treat all others as ports. CJC-Lang treats the two backends as mutually-validating.

## 3. Zero external runtime dependencies

The entire toolchain lives inside the workspace: lexer, parser, type system, both executors, tensor kernels, BLAS-equivalent linear algebra, FFT, statistics, distributions, hypothesis tests, regex, binary serialization, grammar-of-graphics visualization, automatic differentiation, and quantum simulation. No LLVM, no libc math in the hot path, no third-party crates.

This is both a constraint and a deliberate design choice: if the toolchain needs libm to produce `sin(x)`, then `sin(x)` is not deterministic across platforms. Eliminating that dependency is the only way to make the determinism contract honest.

Cost: performance. See [[Performance Profile]] — the tree-walking interpreter is 10–100× slower than native compiled code. CJC-Lang trades speed for reproducibility.

## 4. Numerical stack aimed at scientific workloads

CJC-Lang's built-in surface is shaped like a scientific computing environment, not a general-purpose language:

- ~19+ math functions, ~35+ statistics, 24 distributions (PDF/CDF/PPF), 24 hypothesis tests
- Deterministic linear algebra (matmul, det, solve, lstsq, eigh, svd)
- 40+ ML primitives (relu, gelu, attention, conv2d, Adam)
- FFT/RFFT/IFFT and signal windows
- 73+ DataFrame operations (filter, group_by, join, pivot, window)
- 80 chart types in [[Vizor]]
- Forward-mode and reverse-mode [[Autodiff]]
- A full quantum simulator with VQE, QAOA, MPS, DMRG, stabilizer, density-matrix, and QEC modules ([[Quantum Simulation]])

See [[Builtins Catalog]] for the grounded list.

## 5. The tiered memory model

The [[Memory Model]] has three layers:
1. **Stack-allocated immediate values** (ints, floats, booleans) — no heap involvement.
2. **COW buffers** for tensors and arrays — reference counted, copied on mutation.
3. **RC heap** for everything that escapes, with no mark-sweep cycle collector. Cycles are forbidden by policy; `Weak<T>` is used for back-edges.

The [[MIR Optimizer]] runs an [[Escape Analysis]] pass that classifies allocations as `Stack`, `Arena`, or `Rc` (see `crates/cjc-mir/src/escape.rs`). The [[Frame Arena]] gives per-call bump allocation. The [[NoGC Verifier]] can prove that a function never needs heap allocation at all — this is what makes deterministic microsecond-latency inference tractable in the interpreter.

## 6. The AD-native ethos

Automatic differentiation is not a library bolted on — it's a first-class subsystem in [[cjc-ad]] with:
- Forward-mode via `Dual` numbers
- Reverse-mode via a `ComputeGraph` tape
- A dedicated [[PINN Support]] module for physics-informed neural networks

Partially implemented: MIR-level integration of reverse-mode AD is still on the roadmap (see [[Roadmap]]).

## What this does NOT mean

- CJC-Lang is **not** production-ready. The tree-walking interpreter is slow, the module system is not wired into the default path, and several features (decorators, default params, variadics, browser target) are planned but absent.
- CJC-Lang is **not** claiming novelty on any *single* idea — Kahan summation, Pratt parsing, SSA, COW tensors, SplitMix64, statevector quantum simulation, and grammar-of-graphics are all well-known. What's distinctive is the *combination* held together by the determinism invariant.
- CJC-Lang does **not** compete with PyTorch, NumPy, Julia, or Mathematica on raw capability. It competes with them on *reproducibility discipline* in a much smaller scope.

## Related notes

- [[CJC-Lang Overview]]
- [[Language Philosophy]]
- [[Determinism Contract]]
- [[Numerical Truth]]
- [[Parity Gates]]
- [[Current State of CJC-Lang]]
