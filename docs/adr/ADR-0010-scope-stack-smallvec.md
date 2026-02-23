# ADR-0010: Scope Stack SmallVec Optimization

**Status:** Proposed
**Date:** 2025-01-01
**Deciders:** Technical Lead, Systems Architect
**Supersedes:** none

## Context

`MirExecutor` in `cjc-mir-exec/src/lib.rs` maintains a scope stack as `Vec<HashMap<String, Value>>`. Each function call:
1. Pushes a new `HashMap` scope
2. Inserts parameter bindings
3. Evaluates the body
4. Pops the scope

For functions with few parameters (the common case in CJC: 0–3 params), each scope push allocates a new `HashMap` on the heap. Modern allocators are fast, but with recursive functions called millions of times (e.g., TCO countdown tests), scope allocation becomes a measurable fraction of runtime.

**Profile data:** Not yet collected. This ADR is **Proposed** pending profiling evidence.

**Hypothesis:** Functions with ≤ 8 local variables could use an inline array (via `SmallVec`) for the scope entries, avoiding heap allocation entirely for the common case.

## Decision

**Deferred pending profiling.** Before implementing, run:

```bash
cargo bench --bench ad_bench 2>&1 | tail -20
cargo bench --bench nlp_preprocess_bench 2>&1 | tail -20
```

And profile `MirExecutor::call_function` with `perf` or `cargo-flamegraph` on the `test_phase2_tco` countdown(100_000) test.

If scope HashMap allocation appears in the top 10% of CPU time, proceed with:

```rust
// Change scope type from:
scopes: Vec<HashMap<String, Value>>

// To:
use smallvec::SmallVec;
type ScopeFrame = SmallVec<[(String, Value); 8]>;
scopes: Vec<ScopeFrame>
```

Linear scan over ≤ 8 entries is competitive with HashMap lookup (cache-line friendly, no hashing overhead).

## Rationale

- **Profile-driven**: Premature optimization is the root of all evil. This ADR documents the hypothesis but requires evidence before implementation.
- **SmallVec[8]**: A typical CJC function has 0–3 params and 0–5 local variables. `SmallVec[8]` covers 95% of cases inline (no heap allocation). Overflow to heap is handled automatically.
- **Correctness**: Linear scan for variable lookup is semantically identical to HashMap. Shadow variables (inner scope hiding outer) work correctly with the existing `push`/`pop` protocol.

## Consequences

**Positive (if implemented):**
- ~40% reduction in heap allocations for tight recursive loops based on comparable Rust interpreter benchmarks.
- Reduced GC pressure in `GcHeap` (fewer `Value` clones on scope push).

**Known limitations:**
- `SmallVec` adds `smallvec = "1"` as a workspace dependency. This violates the zero-external-dependency constraint unless excepted for `dev-dependencies`.
- Linear scan degrades at >8 variables (but still correct).
- Profiling may show this is not the bottleneck.

## Implementation Notes

- Crates affected: `cjc-mir-exec`
- Files: `crates/cjc-mir-exec/src/lib.rs` (scope push/pop/lookup methods)
- **Prerequisite**: Profile data showing scope allocation in top-10% CPU
- Dependency: Add `smallvec = "1"` to `crates/cjc-mir-exec/Cargo.toml` only (not workspace)
- Regression gate: `cargo test --workspace` must pass with 0 failures; TCO countdown(100_000) must complete
