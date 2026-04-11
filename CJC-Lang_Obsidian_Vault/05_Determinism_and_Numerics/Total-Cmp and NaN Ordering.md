---
title: Total-Cmp and NaN Ordering
tags: [determinism, numerics]
status: Implemented
---

# Total-Cmp and NaN Ordering

## The problem

`f64::partial_cmp` returns `None` for NaN. That means `sort_by(|a, b| a.partial_cmp(b).unwrap())` panics on NaN inputs, and `sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))` silently reorders NaNs nondeterministically.

Either way, you can't build deterministic sort, argsort, rank, quantile, or median on partial_cmp.

## The solution

`f64::total_cmp` (stable in Rust since 1.62): a total ordering on all f64 values, including signed NaN, following the IEEE 754-2008 totalOrder predicate.

CJC-Lang uses `total_cmp` (and its `f32` counterpart) in every sort-based operation in `crates/cjc-runtime/src/` — stats, sparse eigensolvers, statistics, and the sort builtin.

## Canonical NaN

For serialization and hashing, NaN is canonicalized to a single bit pattern (`0x7FF8_0000_0000_0000`) so that two NaN values produced by different operations hash the same. See [[Value Model]] and [[Binary Serialization]].

## Related

- [[Determinism Contract]]
- [[Numerical Truth]]
- [[Statistics and Distributions]]
- [[Binary Serialization]]
