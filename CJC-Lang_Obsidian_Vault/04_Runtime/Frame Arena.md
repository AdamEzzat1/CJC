---
title: Frame Arena
tags: [runtime, memory]
status: Implemented
---

# Frame Arena

Source: `crates/cjc-runtime/src/frame_arena.rs`.

## Summary

A per-call bump allocator. When a function enters, a fresh arena is created; every `Arena`-classified allocation bumps a pointer inside it. When the function returns, the whole arena is freed at once.

## Why

- **Constant-time allocation** — just a pointer bump.
- **Constant-time deallocation** — just reset the pointer.
- **Deterministic latency** — no `free` fragmentation, no cascading deallocations.
- **Easy reasoning** — an `Arena` value cannot outlive its frame, by construction.

## Who fills the hint

[[Escape Analysis]] in `crates/cjc-mir/src/escape.rs` decides which allocations get `AllocHint::Arena`. Anything that cannot be proven frame-local but is short-lived is a candidate.

## Relationship to NoGC

A function that allocates only `Stack` and `Arena` values is compatible with `@nogc` because no Rc heap churn happens. The [[NoGC Verifier]] uses this to certify functions as allocation-free at the Rc level.

## Related

- [[Memory Model]]
- [[NoGC Verifier]]
- [[Escape Analysis]]
