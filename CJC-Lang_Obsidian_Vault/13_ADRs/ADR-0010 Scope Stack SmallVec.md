---
title: ADR-0010 Scope Stack SmallVec
tags: [adr, proposed, executor, performance]
status: Proposed (deferred pending profiling)
date: 2025-01-01
source: docs/adr/ADR-0010-scope-stack-smallvec.md
---

# ADR-0010 — Scope Stack SmallVec Optimization

**Status:** Proposed (deferred pending profiling) · **Date:** 2025-01-01

## The hypothesis

`MirExecutor` holds its scope stack as `Vec<HashMap<String, Value>>`. For tight recursive calls (think TCO countdown(100_000)), every call allocates a fresh `HashMap` even though most scopes only ever bind 0–3 parameters and 0–5 locals.

**Proposed change:** replace `HashMap<String, Value>` with `SmallVec<[(String, Value); 8]>`. Functions with ≤ 8 local variables get inline storage (no heap allocation); larger scopes spill to the heap automatically. Lookup is a linear scan — cache-friendly and competitive with HashMap at those sizes.

## Why this is still "Proposed"

The ADR is explicit that this is **profile-driven**:

> Premature optimization is the root of all evil. This ADR documents the hypothesis but requires evidence before implementation.

Gate for implementation: `perf` / `cargo-flamegraph` on the countdown TCO test must show scope allocation in the top 10% of CPU time.

## Trade-off

- **Upside (if confirmed):** ~40% fewer allocations in tight recursive loops; less pressure on the GC for captured `Value` clones.
- **Downside:** Adds `smallvec = "1"` as a dependency — a violation of CJC-Lang's zero-external-runtime-dep constraint that would need explicit approval.
- **Risk:** Profile may show scope allocation is *not* the bottleneck, and the change would add a dependency for nothing.

## What this constrains

- Do not implement until profile data justifies it.
- If implemented, `dev-dependencies` should be preferred over workspace dep if feasible.

## Related

- [[cjc-mir-exec]]
- [[Performance Profile]]
- [[ADR Index]]
