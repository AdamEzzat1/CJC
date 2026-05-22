---
title: "ADR-0027: Fused Elementwise Kernels"
tags: [adr, runtime, perf, memory, green, determinism]
status: Accepted (GC-06 Phase 3a shipped)
date: 2026-05-21
---

# ADR-0027: Fused Elementwise Kernels

## Status

**Accepted.** GC-06 Phase 3a (the bounded, thermal-compounding first slice of
the memory-traffic initiative). Adds a small family of single-pass fused
elementwise kernels that eliminate intermediate tensor allocations. Research
context: [[Green Compute Performance Recovery]] (option #1 there). Phase 3b is
the data-DSL group-by/join scaling.

## Context

Plain elementwise expressions allocate an intermediate per operation: `a*b + c`
evaluates as `mul_elem(a,b)` → a fresh tensor, then `add(…, c)` → another. That
is **2 allocations and ~5N words of memory traffic** (read a, read b, write tmp;
read tmp, read c, write out) for an N-element result. Memory traffic is the
dominant energy term and a common bottleneck — and it compounds with the
[[ADR-0026 Race-to-Idle Adaptive Scheduling|race-to-idle scheduler]]: a workload
that moves less data is more likely to finish inside the full-speed burst window
before the thermal cap engages.

A precedent already existed: `Tensor::fused_mul_add` (`a*b + c`, builtin
`broadcast_fma`).

## Decision

Add three more fused single-pass kernels as `Tensor` methods + shared-dispatch
builtins, mirroring `fused_mul_add` exactly:

| Builtin | Computes | Replaces |
|---|---|---|
| `fused_axpy(alpha, x, y)` | `alpha*x + y` (scalar `alpha`) | `scalar_mul` + `add` (BLAS axpy) |
| `fused_mul_sub(a, b, c)` | `a*b - c` | `mul_elem` + `sub` |
| `fused_sub_sq(a, b)` | `(a-b)^2` | `sub` + `mul_elem` (MSE/variance/distance) |

Each: shape-check → contiguous single-pass loop → non-contiguous fallback to the
unfused method sequence. **One allocation, one pass** instead of two.

### Determinism (the load-bearing property)

The fused loops compute e.g. `a[i]*b[i] + c[i]` with **separate** multiply and
add — Rust does *not* contract this into a hardware FMA unless `.mul_add()` is
called explicitly (see [[Float Reassociation Policy]]). So each fused kernel is
**byte-identical** to its unfused sequence (same two roundings, same order). The
memory win is therefore numerically free. Proven by 5 proptests (256 cases each)
asserting `fused == unfused` bit-for-bit, plus a bolero fuzz target.

## Consequences

### Positive
- **~40% less memory traffic and one fewer allocation** per fused expression
  (analytically: ~5N → ~4N words, 2 allocs → 1).
- **Compounds with the adaptive thermal control** — lighter work finishes in the
  burst window, so the cap engages less; you keep more performance.
- **Determinism-neutral** — bit-identical to the unfused path; safe by default,
  no opt-in needed.
- **No new `Value` variant, no new dependency** — pure additive builtins via the
  shared dispatch, so both executors get them and AST↔MIR parity is automatic.

### Negative / limits
- **Explicit, not automatic.** Users / libraries must call the fused builtin; an
  arbitrary `a + b*c` written with plain operators is not auto-fused. (See
  Alternatives — automatic fusion was deferred.)
- **Serial single-pass** (matching `fused_mul_add`). These ops are memory-bound,
  so SIMD/parallelizing them is a possible later refinement, not required for the
  traffic win.

## Alternatives considered

- **A. Lazy tensor expression trees** (defer eval, fuse a whole chain on
  materialization). Rejected for now — a large change to `Value` semantics and
  the executors, with determinism and debugging complexity. Revisit only if
  profiling shows fusion coverage is the bottleneck.
- **B. MIR peephole fusion** (rewrite `add(_, mul(_,_))` → `fused_mul_add`).
  Rejected for now — the *default* executor is `cjc-eval` (no MIR optimization),
  so a MIR-only rewrite would miss the common path. Worth revisiting once the
  Tier-0 work makes the optimized MIR path the default.
- **C. Explicit fused primitives (chosen)** — minimal, determinism-safe, follows
  the `broadcast_fma` precedent, and matches the language's "minimal primitives;
  composition in libraries" rule.

## Tests

`tests/fused_ops/` (17 tests): 5 unit (explicit values, non-contiguous fallback,
shape errors), 5 wiring (AST↔MIR parity + int-alpha + shape-mismatch error
parity), 5 proptest (each fused == unfused, byte-identical, 256 cases; + builtin
== method), 2 bolero fuzz (values match unfused & stay finite; shape mismatch
Errs gracefully and dispatch survives).

## Source

- Branch: `claude/eloquent-lederberg-dad128`
- Crates touched: `cjc-runtime` (`tensor`, `builtins`)

## Related

- [[Green Compute Performance Recovery]] — the option analysis (this is #1)
- [[ADR-0026 Race-to-Idle Adaptive Scheduling]] — what fusion compounds with
- [[Float Reassociation Policy]] — why separate mul+add ≠ FMA (the bit-identity)
- [[Runtime Policy Layer]] — the green-compute umbrella
