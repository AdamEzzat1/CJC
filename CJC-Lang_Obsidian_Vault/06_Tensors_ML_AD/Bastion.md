---
title: Bastion
tags: [runtime, stats, library]
status: Partially implemented (Phase 17 complete)
---

# Bastion

Documented in `docs/bastion/` — a **pure-CJC statistical library** built on top of [[Runtime Architecture]] primitives.

## The 15-primitive model

From `docs/bastion/BASTION_PRIMITIVE_ABI.md`: Bastion operates under a strict rule — admit a new runtime primitive *only if* it unlocks 5–20 higher-level functions that can then be written in pure CJC. The idea is to keep the language core small while making 47% of statistical functionality expressible in user code.

Example: adding `nth_element` (O(n) selection) unlocks the entire family of robust statistics (median, MAD, IQR, quartiles, trimmed mean) which would otherwise require full O(n log n) sorts.

## Classification

`docs/bastion/CLASSIFICATION.md` categorizes ~118 statistical functions:

| Label | Meaning | Count |
|---|---|---|
| **P** | Primitive (in runtime) | 8 |
| **R** | Runtime (not primitive but in core) | 33 |
| **B** | Bastion library (pure CJC) | 55 |
| **X** | Postponed | 15 |
| **!** | Rejected | 1 |

## Determinism contract

`docs/bastion/bastion_determinism_contract.md` spells out the same invariants as [[Determinism Contract]]:
- Identical input + identical seed → identical output bit pattern
- Kahan-compensated summation
- `f64::total_cmp` stable sorting
- `SplitMix64` RNG
- `BTreeMap` iteration order
- Per-primitive bit-identical parity tests

## Status

- The primitives required by Bastion's 15-primitive model have been landing in hardening phases.
- Phase 17 completed the categorical foundations (see below).
- Not every Bastion function is expected to exist as a CJC library yet — it is a direction, not a shipped standalone library.

## Phase 17: categorical foundations

`docs/spec/phase_17_categorical_foundations.md` describes:
- `FctColumn` — u16-indexed factors with explicit level storage (max 65,535 levels)
- `NullableFactor` — factor + bit mask for missing values
- `fct_encode`, `fct_lump` (top-N with "Other"), `fct_reorder`, `fct_collapse`
- 18 spec-locked invariants (S-1 through S-18)
- 66 tests passing + 4 ignored capacity tests

These are foundational for any grouping / categorical workflow, including [[DataFrame DSL]] `group_by`.

## Related

- [[Statistics and Distributions]]
- [[Hypothesis Tests]]
- [[DataFrame DSL]]
- [[Determinism Contract]]
- [[Language Philosophy]]
