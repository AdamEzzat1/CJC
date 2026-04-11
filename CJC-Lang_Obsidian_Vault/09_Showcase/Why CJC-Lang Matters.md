---
title: Why CJC-Lang Matters
tags: [showcase]
status: Evidence-backed
---

# Why CJC-Lang Matters

A candid answer to the "why does this exist?" question.

## The problem

Scientific computing has a reproducibility crisis at the code level, not just the data level. Papers with published seeds and commit hashes still fail to reproduce because:

1. BLAS vendors change rounding across versions.
2. FMA availability differs across CPUs.
3. Thread-pool schedules interact with reduction order.
4. `HashMap` iteration order is randomized for security.
5. libm implementations differ across platforms.
6. Floating-point reassociation by the compiler changes bit patterns.
7. Language-level "seed your RNG" advice doesn't address any of the above.

Each of these is small. Together, they mean that a program that is *correct* and *seeded* still produces different output on a laptop vs. a server vs. a CI runner.

## What CJC-Lang does about it

CJC-Lang tries to **collapse the whole reproducibility stack into one language**:

- [[SplitMix64]] everywhere — no ad-hoc RNG.
- [[Kahan Summation]] and [[Binned Accumulator]] everywhere — no naive `+=` in reductions.
- `BTreeMap` everywhere — no hash order randomness.
- No FMA in SIMD kernels — same bits on FMA and non-FMA hardware.
- No libm in the hot path — CJC-Lang ships its own math.
- No external BLAS — CJC-Lang ships its own linalg.
- Two mutually-validating executors with [[Parity Gates]] — if one drifts, the other catches it.
- Static verification of no-allocation zones — the [[NoGC Verifier]] rules out latency surprises.

## Who it's for

- Researchers who need their numerical results to reproduce exactly across machines.
- ML engineers who want training logs to be byte-identical for debugging.
- Educators teaching floating-point behavior who want a language where rounding is observable and controllable.
- Anyone who finds "scientific notebook fatigue" — environments breaking every six months due to library version drift — and wants a single self-contained toolchain.

## Who it's NOT for (today)

- Large-scale production ML training. Use PyTorch/JAX.
- Performance-critical HPC. Use Julia or C++ with MKL.
- Browser applications. No WASM target yet.
- Deployment at scale. The interpreter is slow; there's no LLVM backend; the module system isn't fully wired.

## The honest positioning

CJC-Lang is **research-grade** and **evidence-backed**. It's the work of a single developer building a serious tool with rigorous discipline. The README, audit docs, and hardening reports are themselves unusually honest about what's implemented vs. planned.

This vault inherits that honesty.

## Related

- [[What Makes CJC-Lang Distinct]]
- [[Language Philosophy]]
- [[Determinism Contract]]
- [[Current State of CJC-Lang]]
- [[Roadmap]]
