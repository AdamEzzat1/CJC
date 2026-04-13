---
title: ADR-0004 SplitMix64 RNG
tags: [adr, accepted, determinism, rng]
status: Accepted
date: 2024-01-15
source: docs/adr/ADR-0004-splitmix64-rng.md
---

# ADR-0004 — SplitMix64 as the Canonical CJC RNG

**Status:** Accepted · **Date:** 2024-01-15

## The decision

**SplitMix64** is the only random number generator in CJC-Lang. Lives in `cjc-repro` as `Rng::seeded(seed: u64)`. `Tensor::randn` uses it with Box-Muller transform for standard normal samples.

## Why this matters

- **Bijective on u64.** Given the same state word, the next state and next output are exactly determined by a constant-time arithmetic function. No float in the state update → no rounding-mode or FMA divergence across platforms.
- **Zero dependencies.** ~8 lines of Rust, fits CJC's zero-runtime-dep constraint.
- **Fast.** ~1 ns/sample is adequate for tensor-scale ML fixtures.
- **Single source of randomness.** Every CJC primitive that uses randomness (`randn`, shuffle, `categorical_sample`, dropout) threads an explicit seed through `Rng`, so the same `--seed` always produces the same bytes.

## Known limits

- **Period 2^64.** Plenty for tests and small-scale ML; too short for cryptography — CJC-Lang makes no crypto guarantees.
- **Box-Muller.** Two float ops per normal sample; slower than Ziggurat but simpler and easier to reproduce across platforms.

## What this constrains

- No crate may introduce a new RNG. If randomness is needed, use `cjc_repro::Rng`.
- New builtins that consume randomness must accept a seed (or inherit from a thread-local `Rng`) and be parity-tested between executors.

## Related

- [[SplitMix64]]
- [[Determinism Contract]]
- [[ADR Index]]
