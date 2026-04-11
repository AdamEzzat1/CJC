---
title: Determinism Contract
tags: [determinism, hub]
status: Implemented (core) / Partially enforced (edges)
---

# Determinism Contract

> *Same source + same seed + same inputs = bit-identical output, on any machine, any thread count, any run.*

This is the top-level invariant that defines CJC-Lang. Everything else — the two executors, the zero-dependency policy, the Kahan summation, the BTreeMap discipline — exists to uphold it.

Grounded in: `docs/bastion/bastion_determinism_contract.md`, `docs/vizor/DETERMINISM.md`, `docs/architecture/byte_first_vm_strategy.md`, CLAUDE.md determinism rules, README.

## The hard rules

1. **All internal maps are `BTreeMap` / `BTreeSet`.** `HashMap`/`HashSet` are essentially banned in compiler and hot-path runtime code because their iteration order is randomized. See [[Deterministic Ordering]].

2. **All floating-point reductions use compensated summation.** Naive `+=` loops are forbidden. Use [[Kahan Summation]] for sequential reductions and [[Binned Accumulator]] for order-independent reductions.

3. **RNG is [[SplitMix64]] with explicit seed threading.** No hidden global state, no `/dev/urandom`, no clock-seeded randomness. Every RNG consumer receives a seed from the top-level CLI `--seed N` flag (default 42).

4. **SIMD kernels do not use FMA (fused multiply-add).** FMA changes rounding (single-rounding vs two-roundings) and produces different bit patterns on hardware with/without FMA support. See [[Float Reassociation Policy]].

5. **Float sorting uses `f64::total_cmp`.** This gives total, deterministic ordering including NaN placement. See [[Total-Cmp and NaN Ordering]].

6. **The MIR optimizer refuses to reassociate floating-point reductions** that carry a reduction annotation. See [[Float Reassociation Policy]] and `crates/cjc-mir/src/reduction.rs`.

7. **Parallel operations must produce identical results regardless of thread count.** (From CLAUDE.md.) Any parallel reduction must use a tree structure with deterministic schedule, not a work-stealing schedule with unpredictable merge order.

8. **NaN is canonicalized in byte form** to `0x7FF8_0000_0000_0000` when serialized, so content-addressable outputs (hashes, [[Binary Serialization]]) are stable.

9. **No libm.** CJC-Lang ships its own `sin`, `cos`, `exp`, `log`, etc., because any dependency on a platform math library is a dependency on that platform's rounding choices. **Needs verification** of the exact policy — some helpers may delegate to Rust's `std` which eventually reaches libm; the spirit is "avoid platform-specific floating-point behavior," not "reimplement everything from scratch."

10. **Parity between executors.** [[cjc-eval]] and [[cjc-mir-exec]] must produce byte-identical output — enforced by [[Parity Gates]].

## Where the contract lives in code

| Concern | Source |
|---|---|
| Kahan / Binned | `crates/cjc-repro/src/` |
| SplitMix64 | `crates/cjc-repro/src/` |
| BTreeMap discipline | across all crates |
| No-FMA SIMD | `crates/cjc-runtime/src/tensor_simd.rs` |
| total_cmp sorting | `crates/cjc-runtime/src/` (stats, sparse_eigen) |
| Reduction contract | `crates/cjc-mir/src/reduction.rs`, `verify.rs` |
| NoGC verification | `crates/cjc-mir/src/nogc_verify.rs` |
| Parity gates | `tests/` |

## Where the contract is weakest

- **Floating-point transcendentals** — `sin`, `cos`, `exp`, `log` in the runtime may delegate to Rust's `std`, which on some platforms reaches libm. This is the *likely* determinism boundary and the hardest to fully close. **Needs verification**.
- **Parallelism** — the project is single-threaded by default; rayon is optional. Parallel kernels exist but must be carefully gated to preserve bit-identical behavior. **Needs verification** of which parallel paths are currently enabled.
- **IO layer** — `file_read`, `json_parse`, and datetime functions depend on the host filesystem and clock. These are obviously nondeterministic, and the contract applies to *pure* computation only.

## Related

- [[Numerical Truth]]
- [[SplitMix64]]
- [[Kahan Summation]]
- [[Binned Accumulator]]
- [[Deterministic Ordering]]
- [[Total-Cmp and NaN Ordering]]
- [[Float Reassociation Policy]]
- [[Parity Gates]]
- [[Determinism Concept Graph]]
