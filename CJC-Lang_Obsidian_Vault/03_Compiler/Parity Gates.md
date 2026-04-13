---
title: Parity Gates
tags: [compiler, testing, determinism]
status: Implemented
---

# Parity Gates

## Summary

A dedicated test suite that enforces the top-level correctness invariant of CJC-Lang: **[[cjc-eval]] and [[cjc-mir-exec]] must produce byte-identical output for any program.** Delivered in Stage 2.4 and continuously expanded.

## The gates

CLAUDE.md and progress docs mention gates like **G-8** and **G-10**. Each gate is a property that must hold for a class of programs:

- Arithmetic parity
- Control-flow parity
- Closure parity (captures, recursion)
- Match / destructuring parity
- Tensor operation parity
- Reduction parity (sum, mean, etc. — see [[Kahan Summation]])
- RNG parity (seed → same sequence in both)
- DataFrame operation parity
- ... (approximately 50+ tests)

## Why parity and not reference comparison to a third tool

Because there is no third tool. CJC-Lang is zero-dependency. The two in-house backends are mutually validating: if one changes behavior, the parity gate catches it and forces a decision about which one is right.

## Process for new features

When a new feature lands, CLAUDE.md prescribes:

1. Add a unit test for the feature.
2. Add a parity test: run it through both executors, assert identical output.
3. Never mark the feature shipped until both gates are green.

## Relationship to determinism

Parity is a *subset* of [[Determinism Contract]] — determinism additionally requires that *each* executor produces the same output on repeated runs with the same seed, across thread counts and platforms. Parity ensures the two executors agree; determinism ensures each is a function only of source + seed.

## Related

- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[Determinism Contract]]
- [[Test Infrastructure]]
- [[Wiring Pattern]]
