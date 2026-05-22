---
title: Green Compute Performance Recovery
tags: [roadmap, perf, green, thermal, determinism, research]
status: Research — Phase 2 (#1 race-to-idle) building; rest staged
---

# Green Compute Performance Recovery

How to recover the throughput a thermal cap costs **without** giving up the
thermal bound. Companion to [[Runtime Policy Layer]] / [[ADR-0025 Runtime Policy Layer]].

## The problem (measured)

Phase 1 shipped a fixed thermal cap. On an 8-core host, a 1024×1024 parallel
matmul (`cjcl bench`, release):

| Profile | Threads | Mean | vs max-perf |
|---|---|---|---|
| `--threads 1` | 1 | 628 ms | 3.0× slower |
| `cool` | 2 | 468 ms | 2.2× slower |
| `balanced` (default) | 4 | 268 ms | 1.3× slower |
| `max-perf` | 8 | 210 ms | baseline |

So the default (`balanced`) costs ~27% on parallel work; `cool` ~2.2×. Results
were **bit-identical** across all of them (MD5 `79aaf99…`) — thread count is a
heat/speed knob, never an answer knob.

## The reframe (why this is solvable)

**Energy = power × time.** Core-capping cuts power but *raises* time — on short
bursts it's often an energy wash (cooler but longer). The levers that recover
performance *and* save energy attack the other terms:

- **Total work** — fewer cycles → less time at the same power.
- **Memory traffic** — the dominant energy term (the [[Runtime Policy Layer|energy model]]
  weights bytes as heavily as FLOPs).
- **When** the thermal budget is spent — *bursts don't overheat; only sustained
  load does* (thermal time constants are seconds-to-tens-of-seconds).
- **Power per core** (frequency / DVFS) instead of *number* of cores.

Core-capping is the only lever that trades performance for heat. The others are
mostly no-tradeoff: they cut time *and* joules. So the redesign shifts the
policy question from "**how many cores?**" to "**how many joules/watts, and
when?**"

## What's already in place (build on, don't rebuild)

- Matmul band-chunking already adapts to the live thread count:
  `band_size = ceil(m / current_num_threads()).max(64)` (`tensor.rs:~900`).
- 6-pass MIR optimizer (CF/SR/DCE/CSE/LICM + SCCP) in `cjc-mir/src/optimize.rs`.
- AVX2 **no-FMA** kernels + 64×64 L2 tiles; COW buffers, frame arena, binned
  allocator, tensor pool.
- Only **4 rayon call sites**, global pool only — small surface to make smarter.

## Options (ranked)

### Tier 1 — recover the burst cost, determinism-safe, no new deps

**#1 Race-to-idle adaptive cap** — *building now (Phase 2).* Run at full
parallelism for short bursts; ramp to the cap only when **sustained** load is
detected (sliding window). Short/interactive runs finish at full speed; only the
long workloads that actually overheat get throttled. Determinism-safe: only the
*schedule* adapts (wall-clock-driven), output stays bit-identical because thread
count ≠ results. `--no-adaptive` gives a fixed (Phase-1) schedule for
reproducible benchmarking. Mechanism: keep the global rayon pool at full size;
throttle by `install`-ing sustained work into a smaller cached pool so the
existing chunkers auto-scale via `current_num_threads()`.

**#2 Hot/cold path split** (roadmap GC-07) — move audit/log/SVG/Merkle/lineage
work off the numerical critical path; `audit_mode` is the hook. Under a cap, cold
work competes harder for the few allowed cores, so deferring it speeds the hot
path *more* than at full width. Determinism-safe (cold output ≠ numeric output).

**#3 Policy-aware thresholds** — the parallel thresholds are hardcoded for an
all-cores world (512×512 matmul; fixed 4096-element SIMD chunk that doesn't adapt
to thread count, `tensor_simd.rs:117`). Make them functions of effective
parallelism; skip rayon entirely when throttled to 1 (serial fast-path). Folded
into #1. Low effort/risk, modest win.

### Tier 2 — reduce total work/traffic (helps every thread count, strictly thermally positive)

**#4 Faster interpreters (Tier-0)** — *separate live initiative; do not entangle.*
`cjcl run` defaults to [[cjc-eval]] with no optimization; the optimizer only runs
under `--mir-opt`, and per [[ADR-0024 Tier-0 Slot Resolution]] the MIR executor is
currently *slower* (lowering + tree-walk overhead) and has an open Stage-5b
regression. Fewer instructions per result = less time *and* less energy with zero
parallelism change — the deepest green lever, but it belongs to
[[Tier-0 Interpreter Perf]], not this initiative.

**#5 Memory-traffic reduction** (roadmap GC-06) — the deferred adaptive-layout +
deterministic-hash-table lever for [[TidyView Architecture]] /
[[ADR-0023 ABNG Adaptive Belief Radix Graph (Phase 0.1)|ABNG]] at scale, plus
making `batch_size` cache-aware (fit L2/L3) rather than thermal-only. Bandwidth
often dominates; less traffic = faster *and* lower-energy. Determinism-safe
(layout ≠ values). Large item, staged.

**#6 Single-core kernel tuning** — register-block the AVX2 AXPY micro-kernel, tune
tile size to actual L2, widen the medium-matmul SIMD use. Helps capped pools for
free (fewer cycles). Must preserve no-FMA + fixed reduction order (parity-gated).
**Stay on AVX2 — AVX-512 down-clocks the core and is thermally counterproductive.**
Fiddly, bit-identity risk; staged.

### Tier 3 — powerful but platform-dependent or result-changing

**#7 Power-budget / DVFS mode** — cap *watts*, not cores: use all cores at a lower
clock (RAPL on Intel/AMD, Windows power plans, Apple Silicon equivalents).
Preserves parallel scaling while bounding power — likely *much* faster than
core-capping for the same thermal envelope, and arguably the *correct* thermal
primitive. **Deliberately not built:** it is platform-specific, often needs
elevated privileges, and most implementations change **global/system** power
settings — a high-blast-radius action a language runtime should not take silently.
Documented as future research requiring a careful, opt-in, process-scoped design.
The `RuntimePolicy` struct is the right home for a future `power_budget` field.

**#8 f32 compute path + `numeric_mode` wiring** — today f32/f16 save *storage* but
all reductions promote to f64. A true f32 compute path doubles AVX2 lane
throughput and halves traffic but **changes results** → must be an explicit
precision mode, not default. Wiring the dormant `numeric_mode` to offer `FixedTree`
(cheaper than Kahan, still deterministic) is a small adjacent win. Staged.

## Memory-efficiency expansion (GC-06)

> **Compounding insight:** memory efficiency and the race-to-idle scheduler
> *multiply*. Adaptive gives full cores for the first ~2s burst; anything that
> makes a workload move less data finishes more of it inside that burst → it
> runs at full speed *and* never trips the cap. Reducing memory traffic doesn't
> just save energy — it expands the set of workloads that stay in the free
> full-speed zone. The two green levers reinforce rather than trade off.

Determinism-safe memory options (memory traffic = energy = heat):

- **Operator/expression fusion** — single-pass kernels that eliminate
  intermediate tensor allocations. **Shipped as GC-06 Phase 3a**
  ([[ADR-0027 Fused Elementwise Kernels]]): `fused_axpy`, `fused_mul_sub`,
  `fused_sub_sq` (joining the existing `broadcast_fma`). Bit-identical to
  unfused; ~40% less traffic, 1 alloc not 2.
- **Finish view/streaming coverage in the data DSL** — filters/selects are
  already lazy + `AdaptiveSelection`-classified and most group-bys stream, but
  `Median`/`Quantile`/`NDistinct`/`First`/`Last` still materialize a `Vec<usize>`
  per group and joins densify. (GC-06 Phase 3b.)
- **Deterministic radix group-by/join** — `DetMap` exists; radix-partitioned
  grouping is cache-friendly (less DRAM traffic) *and* deterministic. (Phase 3b.)
- **Sparse → data path** — CSR/COO exist but only in linalg; keeping a sparse
  filter result sparse downstream avoids densifying.
- **Cache-aware batch/tile sizing** — size to actual L2/L3, not just thermal,
  cutting cache misses.
- **Mixed-precision *storage*** (opt-in) — f16/int8 columns are ½–¼ memory, but
  this **changes results vs f64 storage**, so it must be an explicit precision
  mode, never the default.

## Recommended sequence

1. **#1 race-to-idle adaptive cap** — biggest felt win; returns the burst cost,
   keeps the sustained bound, reuses existing hooks. *(building now)*
2. **#4 Tier-0 + #5 memory-traffic** — no-tradeoff green wins on every thread
   count. #4 lives in [[Tier-0 Interpreter Perf]]; #5 is roadmap GC-06.
3. **#7 power-budget mode** — if a safe process-scoped design exists for the
   target hardware, it may dominate core-capping.
4. **#3 thresholds + #2 hot/cold split** — incremental polish.

## Determinism guardrail (applies to all of the above)

Every option here must preserve the [[Determinism Contract]]: identical output
for identical seed. The enabling fact is that **thread count, chunk count, pool
size, and frequency do not change results** — the reductions use fixed-order
Kahan / [[Binned Accumulator]] summation. Adaptive *scheduling* is therefore
allowed (timing varies run-to-run); adaptive *numerics* are not. Anything that
changes results (e.g. #8 f32) must be an explicit, opt-in, non-default mode.

## Related

- [[Runtime Policy Layer]] — the Phase 1 control surface this extends
- [[ADR-0025 Runtime Policy Layer]] — Phase 1 design
- [[Roadmap]] — the Green Compute (GC-0x) line items
- [[Tier-0 Interpreter Perf]] — the #4 lever's home
- [[Determinism Contract]] — the invariant every option must keep
