---
title: "ADR-0026: Race-to-Idle Adaptive Scheduling"
tags: [adr, runtime, perf, determinism, green, thermal]
status: Accepted (Phase 2 shipped)
date: 2026-05-21
---

# ADR-0026: Race-to-Idle Adaptive Scheduling

## Status

**Accepted.** Phase 2 of the green-compute initiative. Builds on
[[ADR-0025 Runtime Policy Layer]] and **changes its thread-cap mechanism**
(see Decision). Recovers the burst performance the Phase-1 fixed cap cost while
keeping the sustained thermal bound. Research context:
[[Green Compute Performance Recovery]].

## Context

Phase 1's fixed thread cap throttled *all* parallel work uniformly. Measured
cost (8-core host, 1024×1024 matmul, release): `balanced` (½ cores) ~27% slower
than `max-perf`, `cool` (¼ cores) ~2.2× slower. Results were bit-identical
across caps — **thread count is a heat/speed knob, never an answer knob.**

The key observation: **bursts don't overheat; only sustained load does.** CPU
thermal time constants are seconds-to-tens-of-seconds, so a sub-second burst at
full width doesn't heat-soak the package. The Phase-1 cap paid the throttle cost
on *every* op, including the short/interactive runs that never threatened to
throttle. We want full width for bursts and the cap only for sustained load.

This is safe to make adaptive *because* of the proven invariant: concurrency
(thread count, chunk count, pool size) does not change results.

## Decision

### Race-to-idle: burst full, throttle when sustained

`run_parallel` decides per-op whether to throttle. A burst timer
([`decide_sustained`], a pure function unit-tested with explicit instants):

- Full width while the current burst has been active < `SUSTAIN_WINDOW` (2 s).
- Throttle to the thermal cap once continuous activity passes the window.
- An idle gap > `IDLE_RESET` (500 ms) starts a fresh burst (low duty cycle =
  low heat → full speed again).

`--no-adaptive` (and the `adaptive` policy field / `runtime_policy_set_adaptive`
builtin) forces the cap uniformly — a fixed, reproducible schedule for
benchmarking.

### Pool model change (supersedes ADR-0025's mechanism)

ADR-0025 shrank rayon's **global** pool to the cap (`build_global(cap)`). That
makes bursts impossible (the pool is small) and can't change at runtime
(`build_global` is once-only). Phase 2 instead:

- Leaves the **global pool at full size** (all cores) so bursts use everything.
- Throttles by `install`-ing sustained work into a **cached cap-sized pool**
  (`OnceLock<rayon::ThreadPool>`). Inside `install`,
  `rayon::current_num_threads()` reports the cap, so the existing band/row/chunk
  splitters **auto-scale** — no per-site chunk-math changes.

```rust
pub fn run_parallel<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() { return work(); } // nested → inline
    let cap = effective_threads(&get(), detect_cores());
    if cap >= detect_cores() { return work(); }                   // no cap
    let throttle = if get().adaptive { is_sustained_now() } else { true };
    if !throttle { return work(); }                               // burst → global
    match capped_pool(cap) { Some(p) => p.install(work), None => work() }
}
```

The four parallel kernels (large matmul, medium matmul, batch matmul, SIMD
binop) wrap their bodies in `run_parallel`. A re-entrancy guard
(`current_thread_index`) prevents nested `install`.

### Determinism

`run_parallel`'s choice (burst vs throttle, which pool) is wall-clock-driven, so
the **schedule is non-deterministic** run-to-run. But output is **bit-identical**
because every wrapped kernel is concurrency-invariant: matmul bands and rows are
disjoint with a fixed within-row reduction order ([[Binned Accumulator]] /
Kahan), and the SIMD binop is elementwise. Adaptive *scheduling* is allowed;
adaptive *numerics* are not. See [[Determinism Contract]].

## Consequences

### Positive
- **Recovers burst performance** — short/interactive runs (the common case) run
  at full width; only genuinely sustained workloads throttle.
- **Keeps the sustained thermal bound** — multi-second load still caps to the
  thermal profile.
- **Auto-covers all parallel sites** inside a wrapped kernel (throttle is pool-
  based, not per-chunk), so future rayon code inside those ops inherits it.
- **No new dependencies**; rayon stays an optional `cjc-runtime` feature.

### Negative / limits
- **Benchmarks need `--no-adaptive`** for a reproducible schedule (timing varies
  run-to-run; results never do).
- **A single op longer than `SUSTAIN_WINDOW`** runs at the width chosen when it
  *started* — throttling is decided between ops, not mid-kernel. Acceptable:
  most workloads are many ops, not one giant op.
- **The capped pool adds ~`cap` mostly-sleeping threads** and `OnceLock` fixes
  its size at the first cap seen (the startup cap; runtime `set_threads` stays
  advisory, as in ADR-0025).

### Risk register
- **New parallel sites must wrap in `run_parallel`.** Because the global pool is
  now full (not shrunk), an unwrapped `par_iter` runs at full concurrency even
  in `cool` mode — a thermal leak. The 4 current sites are wrapped; this ADR
  documents the rule for future ones.

## Alternatives considered

- **A. Route chunk-count through a policy value at each site.** Rejected — must
  edit the hot chunk math at every site and *every* site must be found or it
  leaks heat; the pool-`install` approach throttles all work inside a wrapped op
  for free.
- **B. Resize the global pool adaptively.** Rejected — rayon's global pool is
  configured once (`build_global`) and cannot resize.
- **C. Keep Phase-1 global-pool shrinking.** Rejected — a small global pool can't
  burst above the cap, which is the entire point.

## Tests

- Unit (`runtime_policy.rs`): `decide_sustained_burst_then_throttle`,
  `decide_sustained_idle_resets_burst` (explicit instants), `set_adaptive_round_trip`,
  `run_parallel_preserves_value_under_throttle`.
- Integration (`tests/runtime_policy/wiring.rs`): adaptive builtins AST↔MIR
  parity, bad-type rejection, and **`adaptive_does_not_change_matmul_results`** —
  a 256×256 parallel matmul byte-identical across adaptive on/off and both
  executors.
- CLI: a 256×256 matmul hashed identical across `--no-adaptive`, profiles, and
  both executors.

## Source

- Branch: `claude/eloquent-lederberg-dad128`
- Crates touched: `cjc-runtime` (`runtime_policy`, `tensor`, `tensor_simd`,
  builtins), `cjc-cli` (`--no-adaptive`)

## Related

- [[ADR-0025 Runtime Policy Layer]] — Phase 1; this changes its cap mechanism
- [[Runtime Policy Layer]] — the concept note
- [[Green Compute Performance Recovery]] — the full option analysis (#1 = this)
- [[ADR-0011 Parallel Matmul]] / [[Binned Accumulator]] — why concurrency ≠ results
- [[Determinism Contract]] — the invariant preserved
