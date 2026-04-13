---
title: Escape Analysis
tags: [compiler, memory, analysis]
status: Implemented
---

# Escape Analysis

Source: `crates/cjc-mir/src/escape.rs`.

## Summary

Intraprocedural escape analysis that classifies allocations with an `AllocHint`:

- **`Stack`** — value lives entirely within a frame; no heap involvement.
- **`Arena`** — value is local to a frame but too big or dynamic for the stack; use [[Frame Arena]].
- **`Rc`** — value may escape the frame; use reference-counted heap allocation.

## Conservative defaults

- Primitives (ints, floats, small fixed values): always `Stack`.
- Intermediate aggregates with no observed escape: optimistically `Arena`.
- Anything that flows into a return, a captured closure, a global, or a boxed destination: downgraded to `Rc`.

The analysis is intentionally conservative: if evidence of escape is present, the hint drops from `Arena` or `Stack` down to `Rc`. It never incorrectly claims a value is stack-local when it actually escapes.

## Feeds into

- [[NoGC Verifier]] — a function is `may_gc` if it contains an `Rc` allocation.
- [[cjc-mir-exec]] — uses hints to pick the right allocator (stack slot vs. frame arena vs. Rc heap).
- [[Memory Model]] — implements the three-tier system the hints describe.

## Related

- [[NoGC Verifier]]
- [[Memory Model]]
- [[Frame Arena]]
- [[MIR]]
