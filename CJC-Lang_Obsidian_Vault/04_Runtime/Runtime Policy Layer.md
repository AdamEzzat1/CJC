---
title: Runtime Policy Layer
tags: [runtime, perf, determinism, green, energy, thermal]
status: Phase 1 shipped (struct + thermal profiles + thread cap + CLI + builtins + tests)
---

# Runtime Policy Layer

The **green-compute control surface** for CJC-Lang. It lets a run declare
*how much machine* it is willing to use — thread caps, thermal profiles,
batch sizing, audit depth — and exposes a **deterministic** energy estimate
so a program can reason about *joules per result* instead of merely
wall-clock seconds.

Lives in `crates/cjc-runtime/src/runtime_policy.rs`, a thread-local sink that
mirrors [[Wiring Pattern|profile.rs]]. Design decision: [[ADR-0025 Runtime Policy Layer]].

## Why this exists

CJC-Lang's whole direction is *efficient trustworthy computation* rather than
brute-force throughput: deterministic, bounded, auditable, memory-aware,
numerically stable. The Runtime Policy Layer makes the *bounded* part
explicit. Before it, parallel kernels ran on rayon's default global pool (all
cores), so a laptop training loop would sustain turbo across every core and
thermally throttle.

The principle:

> Do not let CJC blindly saturate the CPU. Use deterministic bounded
> execution. Make thermal/energy limits explicit and deterministic rather
> than relying on the OS.

This is safe **only because thread count never changes numeric output** —
the parallel reductions use a fixed chunk order with Kahan / binned
summation ([[ADR-0011 Parallel Matmul]], [[Binned Accumulator]]). A thread
cap moves the speed/heat axis, never the answer axis. See [[Determinism Contract]].

## The policy

```rust
pub struct RuntimePolicy {
    pub determinism: Determinism,   // Strict (shipped) | Relaxed (reserved)
    pub numeric_mode: NumericMode,  // Kahan | Binned | FixedTree
    pub thermal_mode: ThermalMode,  // Cool | Balanced | MaxPerf
    pub max_threads: usize,         // 0 = auto (from thermal_mode)
    pub batch_size: usize,          // advisory, chunked workloads
    pub audit_mode: AuditMode,      // Summary | Full | Forensic
}
```

The default is **`Balanced`**, deliberately not "max all cores forever".

### Thermal profiles

| Profile | Threads | Batch | Audit | Use |
|---|---|---|---|---|
| `cool` | ≈ ¼ cores | 32 | summary | gentle on a laptop, edge devices |
| `balanced` (default) | ≈ ½ cores | 128 | full | normal operation, leaves headroom |
| `max-perf` | all cores | 512 | summary | benchmarking, minimal audit overhead |

`effective_threads(policy, cores)` is a pure function returning a value in
`1..=cores`. An explicit `--threads N` wins (clamped to cores); otherwise the
thermal mode derives the cap.

## Energy model — deterministic, workload-only

```
joules ≈ flops × ENERGY_PER_FLOP_JOULES + bytes × ENERGY_PER_BYTE_JOULES
       (~100 pJ/flop)                     (~100 pJ/byte DRAM)
```

The estimate is a pure function of **workload counts**, never wall-clock
time — so it is bit-identical across runs (same program + seed → same FLOP
count → same joules). The two multiplies are kept in separate bindings so
the compiler cannot fuse them into an FMA ([[Float Reassociation Policy]]).

The absolute joules are an order-of-magnitude *estimate*; the **ratio**
between two CJC-Lang runs is the meaningful, marketable number. This is what
makes a metric like *trustworthy-results-per-joule* — `(reproducible,
calibrated, useful output) / energy consumed` — expressible:

```cjcl
// joules per prediction, computed in pure CJC-Lang
let j_total = energy_estimate(total_flops, total_bytes);
let j_per_pred = j_total / n_predictions;
```

`energy_estimate` (and `energy_per_flop` / `energy_per_byte`) are
**deterministic and safe to use in program logic**. The primitive stays
minimal — higher-level metrics (`AUC_per_joule`, `trust_adjusted_*`) belong
in user code / [[Bastion]], per the language's "minimal primitives" rule.

## Builtin surface (15)

Routed through the shared `dispatch_builtin`, so both [[cjc-eval]] and
[[cjc-mir-exec]] get them for free ([[Wiring Pattern]]).

| Builtin | Returns | Deterministic? |
|---|---|---|
| `runtime_policy_thermal_mode()` | String | yes (policy state) |
| `runtime_policy_set_thermal_mode(s)` | String | yes |
| `runtime_policy_threads()` | Int | **no — observability only** |
| `runtime_policy_set_threads(n)` | Int | no (returns effective) |
| `runtime_policy_batch_size()` | Int | yes |
| `runtime_policy_set_batch_size(n)` | Int | yes |
| `runtime_policy_audit_mode()` | String | yes |
| `runtime_policy_set_audit_mode(s)` | String | yes |
| `runtime_policy_numeric_mode()` | String | yes |
| `runtime_policy_set_numeric_mode(s)` | String | yes |
| `runtime_policy_adaptive()` | Bool | yes |
| `runtime_policy_set_adaptive(b)` | Bool | yes |
| `runtime_policy_reset()` | Int (0) | yes |
| `runtime_policy_summary()` | String | **no — observability only** |
| `energy_estimate(flops, bytes)` | Float | **yes — safe in logic** |
| `energy_per_flop()` | Float | yes |
| `energy_per_byte()` | Float | yes |

### Observability contract

`runtime_policy_threads` and `runtime_policy_summary` depend on the host core
count. Like the [[Wiring Pattern|profiler]] handles, **they must not feed
program logic, RNG draws, or output-affecting control flow** — doing so makes
output machine-dependent and breaks cross-platform determinism. The
`energy_estimate*` builtins are pure and carry no such restriction.

## CLI surface

```bash
cjcl run model.cjcl --profile cool       # ¼ cores when sustained, gentle
cjcl run model.cjcl --profile balanced   # default — ½ cores when sustained
cjcl bench model.cjcl --profile max-perf # all cores, benchmark
cjcl run model.cjcl --threads 2          # explicit cap (overrides profile)
cjcl run model.cjcl --profile cool --batch-size 16 --audit summary
cjcl bench model.cjcl --profile cool --no-adaptive  # fixed cap (reproducible)
```

Applied once at the top of `cli_main`, before any command runs, so the
thermal/thread bound covers `run`, `bench`, and every other command.
`--profile` is applied first so explicit `--threads` / `--batch-size` /
`--audit` overrides win over the presets. See [[CLI Surfaces]].

## Phase 2 — race-to-idle adaptive scheduling

Design: [[ADR-0026 Race-to-Idle Adaptive Scheduling]]. By default (`adaptive`,
on for `cool`/`balanced`) parallel work runs at **full width for a short burst**
and only **throttles to the cap once load is sustained** (~2 s window). Short or
interactive runs finish at full speed; only the long workloads that actually
overheat get capped. `--no-adaptive` applies the cap uniformly (a fixed,
reproducible schedule — use it for benchmarking).

Mechanism: the **global rayon pool stays at full size** (so bursts can use all
cores); throttling `install`s sustained work into a cached cap-sized pool, inside
which the existing band/row/chunk splitters auto-scale via
`rayon::current_num_threads()`. The four parallel kernels (large/medium/batch
matmul, SIMD binop) wrap their bodies in `runtime_policy::run_parallel`.
**Determinism holds:** the burst/throttle choice changes only how many bands run
concurrently, never the per-element math — verified by
`adaptive_does_not_change_matmul_results` (256×256 matmul byte-identical across
adaptive on/off and both executors). The *schedule* is non-deterministic
(wall-clock-driven); the *output* is not.

## Tests (54)

- **16 unit** — `cjc-runtime/src/runtime_policy.rs` (presets, energy
  monotone/additive/non-negative, `effective_threads` bounds, enum round-trips).
- **27 wiring** — `tests/runtime_policy/wiring.rs` — every builtin byte-equal
  across both executors ([[Parity Gates]]).
- **8 proptest** — `tests/runtime_policy/proptest.rs` — energy properties,
  mode round-trips, unknown-mode rejection, batch round-trip + clamp.
- **3 bolero fuzz** — `tests/runtime_policy/fuzz.rs` — dispatch survives
  arbitrary op sequences / mode strings / energy inputs.

## Why it could make CJC-Lang greener

The argument is *efficient trustworthy computation*, not "green AI magic":

- **Deterministic replay reduces wasted retraining** — `same seed → same
  output` ([[SplitMix64]]) eliminates the "why did the run change?" reruns
  that dominate ML waste.
- **Numerical stability reduces retries** — Kahan/binned/fixed-tree
  reductions cut NaN/exploding runs and silent drift, so fewer jobs fail and
  rerun.
- **Thermal profiles prevent wasteful saturation** — laptop/edge workloads
  avoid turbo oscillation and throttling.
- **Memory-efficient layouts cut energy** — memory traffic is a dominant
  energy consumer; [[TidyView Architecture]]'s sparse gathers + dictionary
  encoding ([[ADR-0018 Deterministic Adaptive Dictionary Engine]]) reduce
  bandwidth, cache misses, and allocations.

Honest scope: CJC-Lang will not beat hyperscale GPU clusters on raw
throughput. The niche is **dramatically more efficient per *trustworthy*
result** — scientific, medical, regulated, reproducible, edge workloads where
calibration, determinism, and auditability matter more than benchmark TFLOPs.

## Roadmap / deferred levers

Phase 1 (this ADR) shipped the policy, profiles, thread cap, energy estimate,
CLI, and tests. The full performance-recovery analysis (8 ranked options + the
energy=power×time reframe) lives in [[Green Compute Performance Recovery]].
Sequenced follow-ups:

1. **Wire `numeric_mode` into the live reduction dispatch** — today it is a
   recorded preference, not yet enforced on summation.
2. **Instrumented FLOP/byte counter** — a deterministic op counter so whole
   programs report energy automatically (today the program supplies the
   counts to `energy_estimate`).
3. **Adaptive layout abstraction + deterministic hash-table** — the third
   system-engineering lever from the review: memory + scaling headroom for
   the *next* dataset size, unblocking Phase 1.0 of [[TidyView Architecture]]
   / [[ADR-0023 ABNG Adaptive Belief Radix Graph (Phase 0.1)|ABNG]] at real scale.
   Memory traffic is energy, so this is squarely part of the green story.
4. **Hot/cold path split** — keep the fast numerical path (training,
   inference, tensor math) separate from the expensive cold path (audit logs,
   SVGs, Merkle trees) so the machine is not doing everything at once. The
   `audit_mode` field is the hook; the split itself is future work.
5. **Energy/perf telemetry** — `joules_per_epoch`, `AUC_per_joule`, and a
   `trust_adjusted_auc_per_joule` headline metric, built in user code /
   [[Bastion]] on top of `energy_estimate`.

## Related

- [[ADR-0025 Runtime Policy Layer]] — the design decision
- [[Runtime Architecture]] — where this sits in the runtime
- [[ADR-0011 Parallel Matmul]] — why thread count is determinism-neutral
- [[Determinism Contract]] — the invariants preserved
- [[Wiring Pattern]] — one dispatch arm, both executors
- [[TidyView Architecture]] — beneficiary of the memory-layout lever
- [[Performance Profile]] — broader perf picture
