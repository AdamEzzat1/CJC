---
title: ADR-0009 Vec COW Array
tags: [adr, proposed, runtime, memory]
status: Proposed
date: 2025-01-01
source: docs/adr/ADR-0009-vec-cow-array-tuple.md
---

# ADR-0009 — `Rc<Vec<Value>>` for `Value::Array` and `Value::Tuple`

**Status:** Proposed · **Date:** 2025-01-01

## The decision (proposed)

Change the runtime value representation from:

```rust
Array(Vec<Value>),
Tuple(Vec<Value>),
```

to:

```rust
Array(Rc<Vec<Value>>),
Tuple(Rc<Vec<Value>>),
```

Mutation sites use `Rc::make_mut(&mut v)` — the underlying buffer is cloned **only when a mutator discovers another handle still holds a reference** (classic copy-on-write).

## Why this matters

- **Pass-by-value becomes O(1).** Today, every function argument, return value, pattern bind, or `let b = a` clones the full vector. With `Rc`, the clone is an `Rc::clone` — one refcount bump.
- **Mutation stays correct.** `Rc::make_mut` transparently deep-copies when `strong_count > 1`, so programs that mutate aliased arrays still observe normal value semantics.
- **Scope of impact.** Audit found ~98 match sites across 6 crates (runtime, eval, mir-exec, types, hir, data). Most are read-only (`if let Array(ref v)`) and compile unchanged via `Deref`. Only construction (`Value::Array(vec![...])`) and mutation sites need edits.

## Why "Proposed" and not yet accepted

This migration touches two executors and must not break parity. The ADR exists so the change has a single rationale to point at when the migration is finally scheduled — ideally bundled with a future runtime module split (referenced in the source ADR as ADR-0008, not present in `docs/adr/`) so the construction sites get touched once.

## Known limits

- `Rc::make_mut` still copies on COW trigger, so a hot path that mutates many aliased arrays may see no improvement.
- `Rc` is not `Send`. Acceptable because CJC-Lang is single-threaded at the value level.

## What this constrains

- The parity gate `milestone_2_4/parity` is the acceptance test.
- New audit test `test_audit_cow_array.rs` must validate COW semantics (mutation through one handle does not visibly affect the other).

## Related

- [[COW Buffers]]
- [[Value Model]]
- [[Parity Gates]]
- [[ADR Index]]
