---
title: SplitMix64
tags: [determinism, rng]
status: Implemented
---

# SplitMix64

**Source**: `crates/cjc-repro/src/` (primary), with a local copy in `crates/cjc-quantum/src/lib.rs` to avoid cross-crate coupling.

## Summary

A simple, fast, fully-deterministic 64-bit PRNG. Output depends only on the seed and the number of draws, never on wall clock or thread identity.

## The core step

```rust
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub fn rand_f64(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
```

The constants are the published SplitMix64 constants. The uniform `[0,1)` conversion uses 53 bits to fill the f64 mantissa exactly.

## Why SplitMix64

- **Deterministic by construction** — pure function of `state`.
- **Fast** — a few cycles per draw, no lookup tables.
- **Well-distributed** — passes standard statistical tests (PracRand, BigCrush in reasonable regimes).
- **Seedable per-call** — you can thread a separate seed through any subcomputation without touching global state.

## Seed threading

The top-level seed comes from `cjcl run --seed N` (default 42). It flows into:

- [[cjc-eval]] via `Interpreter::new(seed)`
- [[cjc-mir-exec]] via `run_program_with_executor(&program, seed)`
- [[cjc-runtime]] distributions (`normal_ppf`, `t_ppf`, sampling)
- [[cjc-quantum]] measurement sampling
- [[cjc-data]] for random row operations

## Related

- [[cjc-repro]]
- [[Determinism Contract]]
- [[Numerical Truth]]
- [[Quantum Simulation]]
