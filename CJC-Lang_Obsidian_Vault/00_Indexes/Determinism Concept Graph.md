---
title: Determinism Concept Graph
tags: [concept-graph, determinism]
status: Graph hub
---

# Determinism Concept Graph

How determinism is achieved in CJC-Lang — a web of invariants, not a single switch.

## The contract

[[Determinism Contract]] is the top-level statement. It imposes ten hard rules. Each rule is enforced by one or more primitives below.

## Randomness

[[SplitMix64]] is the only RNG.

- Seeds flow explicitly through every function that needs randomness.
- `categorical_sample`, `q_sample`, and RL rollouts all thread seeds.
- No global RNG. No thread-local RNG. No OS entropy.

## Reductions

[[Kahan Summation]] and [[Binned Accumulator]] are the only two permitted summation strategies.

- Kahan: fixed order, O(1) extra state, compensated.
- Binned: commutative, order-independent, used for parallel paths.

Neither uses FMA. See [[Float Reassociation Policy]].

## Ordering

[[Deterministic Ordering]] requires:

- `BTreeMap` / `BTreeSet` only — no `HashMap` / `HashSet`.
- [[Total-Cmp and NaN Ordering]] — `f64::total_cmp` is the only sort comparator.

## Float reassociation

[[Float Reassociation Policy]] forbids:

- FMA instructions in SIMD kernels.
- Reassociation of `(a+b)+c` to `a+(b+c)` by the compiler.
- `-ffast-math` equivalents.

## Numerical truth

[[Numerical Truth]] explains the five stability strategies: Kahan, binned, iterative refinement, compensated arithmetic, and explicit NaN/Inf policies.

## Serialization

[[Binary Serialization]] canonicalizes NaN bit patterns before hashing so that two runs producing different NaN representations still hash identically.

## Parity

[[Parity Gates]] enforce the cross-executor invariant: every valid program must produce byte-identical output in [[cjc-eval]] and [[cjc-mir-exec]]. This is the **ultimate** determinism test — any drift between the two executors is caught immediately.

## Why determinism matters

See [[Why CJC-Lang Matters]] for the motivation: scientific reproducibility that persists across machines, vendors, and time.

## What determinism costs

- No thread-pool reductions with adversarial schedules.
- No vendor BLAS (loses hand-tuned speed).
- No FMA (loses ~2× on some kernels).
- No HashMap iteration order (loses insertion-order shortcuts).

CJC-Lang accepts these costs because the alternative is irreproducibility.

## Related

- [[Determinism Contract]]
- [[Numerical Truth]]
- [[Runtime Concept Graph]]
- [[CJC-Lang Knowledge Map]]
