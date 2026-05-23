---
title: "ADR-0025: Runtime Policy Layer (Green Compute)"
tags: [adr, runtime, perf, determinism, green]
status: Accepted (Phase 1 shipped)
date: 2026-05-20
---

# ADR-0025: Runtime Policy Layer (Green Compute)

## Status

**Accepted.** Phase 1 shipped: the `RuntimePolicy` struct (six fields),
three thermal profiles (`cool` / `balanced` / `max-perf`), a rayon thread
cap applied at CLI startup, four CLI flags (`--profile`, `--threads`,
`--batch-size`, `--audit-mode`), 15 policy + energy builtins routed through the
shared dispatch, and a four-flavor test suite (16 unit + 27 wiring + 8
proptest + 3 bolero fuzz = 54 tests).

Deferred to a later phase (documented under [[Runtime Policy Layer]] →
Roadmap): wiring `numeric_mode` into the live reduction dispatch, an
instrumented FLOP/byte counter for whole-program energy accounting, and
the third system-engineering lever — an **adaptive layout abstraction +
deterministic hash-table** that gives [[TidyView Architecture]] / [[ADR-0023 ABNG Adaptive Belief Radix Graph (Phase 0.1)|ABNG]] memory + scaling headroom at the next dataset size.

## Context

CJC-Lang's direction — deterministic, bounded, auditable, memory-aware,
numerically stable — naturally pushes toward *efficient trustworthy
computation* rather than brute-force throughput. But nothing in the runtime
bounded *how much machine* a run was allowed to use: parallel kernels ran on
rayon's default global pool (all cores), and a laptop running a training loop
would sustain turbo across every core and thermally throttle.

The guiding principle from the systems review:

> Do not let CJC blindly saturate the CPU. Use deterministic bounded
> execution. Make thermal/energy limits explicit and deterministic rather
> than relying on the OS.

This is only *safe* because of an existing invariant: **thread count never
changes numeric output**. The parallel matmul / reduction kernels (see
[[ADR-0011 Parallel Matmul]], [[Binned Accumulator]]) reduce over a fixed
chunk order with Kahan / binned summation, so the result is bit-identical
regardless of how many rayon workers are live. A thread cap is therefore a
pure performance/heat knob — it moves the speed axis, never the answer axis.

The second half of the story is a deterministic **energy estimate**. A
wall-clock-based energy figure would vary run-to-run and poison determinism
(Prime Directive #3). The estimate must be a pure function of the *workload*.

## Decision

Add a **Runtime Policy Layer** in `cjc-runtime` — a single module
`runtime_policy.rs` mirroring the [[Wiring Pattern|profile.rs]] thread-local
sink — that owns the policy state, the thermal presets, the thread
resolution, and the energy model.

### The policy struct + mode enums

```rust
pub struct RuntimePolicy {
    pub determinism: Determinism,   // Strict (only shipped mode) | Relaxed (reserved)
    pub numeric_mode: NumericMode,  // Kahan | Binned | FixedTree
    pub thermal_mode: ThermalMode,  // Cool | Balanced | MaxPerf
    pub max_threads: usize,         // 0 = auto (derive from thermal_mode)
    pub batch_size: usize,          // advisory, for chunked workloads
    pub audit_mode: AuditMode,      // Summary | Full | Forensic
}
```

Each enum has `as_str()` / `from_str()` for the CLI and the builtins. The
default is **`Balanced`**, not "max all cores forever" — laptop-safe by
default.

### Thermal presets (the headline green knob)

| Mode | Threads | Batch | Audit |
|---|---|---|---|
| `cool` | ≈ ¼ cores | 32 | summary |
| `balanced` (default) | ≈ ½ cores | 128 | full |
| `max-perf` | all cores | 512 | summary |

Resolution is a pure function — `effective_threads(policy, detected_cores)`
returns a value in `1..=cores`. An explicit `--threads N` wins (clamped to
the detected cores); otherwise the thermal mode derives the cap.

### Energy model — deterministic, workload-only, no FMA

```rust
pub const ENERGY_PER_FLOP_JOULES: f64 = 1.0e-10;  // ~100 pJ/flop, representative
pub const ENERGY_PER_BYTE_JOULES: f64 = 1.0e-10;  // ~100 pJ/byte DRAM traffic

pub fn energy_estimate_joules(flops: i64, bytes: i64) -> f64 {
    let flop_energy = flops.max(0) as f64 * ENERGY_PER_FLOP_JOULES;
    let byte_energy = bytes.max(0) as f64 * ENERGY_PER_BYTE_JOULES;
    flop_energy + byte_energy   // separate bindings — no FMA contraction
}
```

Wall-clock time is **not** an input. Same program + same seed → same FLOP
count → same joule estimate, bit-for-bit. The two multiplies are bound to
separate `let`s so the compiler cannot contract them into a fused
multiply-add (the same no-FMA discipline the SIMD kernels follow — see
[[Float Reassociation Policy]]). The absolute joules are an *estimate*; the
*ratio* between two CJC-Lang runs is the meaningful, marketable number
("trustworthy results per joule").

### Applying the thread cap to rayon

rayon's global pool can only be configured once per process, so:

```rust
#[cfg(feature = "parallel")]
pub fn apply_thread_cap(n: usize) -> usize {
    static APPLY: Once = Once::new();
    APPLY.call_once(|| {
        if n > 0 { let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global(); }
    });
    rayon::current_num_threads()
}
```

The CLI calls this once at the top of `cli_main`, before any command runs,
so the cap is in effect for `run`, `bench`, and everything else. With no
flags it enforces the resolved `Balanced` cap. Best-effort: if a pool already
exists the `build_global` error is intentionally ignored. Library/test code
never calls it, so the shared test-process rayon pool is untouched.

### Builtins — one dispatch arm, both executors for free

The 15 builtins (`runtime_policy_*` + `energy_estimate` / `energy_per_flop`
/ `energy_per_byte`) are **stateless** dispatch arms in
`cjc_runtime::builtins::dispatch_builtin`. Because both [[cjc-eval]] and
[[cjc-mir-exec]] call `dispatch_builtin` first in their chain, registering
once auto-wires both — and AST↔MIR parity is almost free. This is the
[[Wiring Pattern]] working exactly as the profiler builtins demonstrate.

### CLI flags

`--profile <mode>`, `--threads <N>`, `--batch-size <N>`, `--audit-mode <mode>`
are parsed and applied process-wide in `cli_main`; `--profile` is applied
first so explicit field overrides win over the profile presets. For
CLI-suite subcommands (e.g. `bench`) the flags are stripped from `sub_args`
after application, so subcommands neither need nor choke on them.

## Consequences

### Positive

- **Laptop-safe by default** — `cjcl run` now uses ≈ half the cores rather
  than saturating every core, reducing sustained heat and turbo throttling.
- **Deterministic energy metric** — programs can compute joules-per-result
  without breaking reproducibility; the figure is a pure function of the
  workload.
- **Single dispatch point auto-wires both executors** — no `cjc-eval` /
  `cjc-mir-exec` edits, parity nearly free (proved by 27 wiring tests).
- **No new `Value` variant, no struct breakage** — the policy lives in a
  thread-local; handles are plain `Int`/`String`/`Float`. Nothing in the
  public MIR/Value surface changed (contrast [[ADR-0024 Tier-0 Slot Resolution]]'s breaking field add).
- **Determinism unchanged** — thread count and thermal mode never alter
  numeric output; the energy estimate is wall-clock-free.

### Negative

- **Default behavior change** — `cjcl run` defaults to a half-core cap.
  Intentional (the systems review asked for a balanced default), but it
  means raw single-invocation throughput drops unless `--profile max-perf`
  is passed. `cjcl bench` users who want all cores must opt in.
- **Energy constants are estimates** — `ENERGY_PER_FLOP/BYTE_JOULES` are
  order-of-magnitude representative figures, not calibrated for any chip.
  Documented as such; ratios are meaningful, absolutes are not.
- **Query builtins are machine-dependent** — `runtime_policy_threads` and
  `runtime_policy_summary` depend on the host core count. Like the profiler
  handles, they carry an **observability contract**: must not feed program
  logic, RNG, or output-affecting control flow, or cross-platform
  determinism breaks. The `energy_estimate*` builtins *are* deterministic
  and safe in logic.

### Risk register

- A future caller might branch on `runtime_policy_threads()` and silently
  make output machine-dependent. Mitigation: the contract is documented at
  the dispatch site and in [[Runtime Policy Layer]]; the wiring tests assert
  only parity (same machine), never an absolute value.
- `numeric_mode` is currently stored but not yet wired into the live
  reduction dispatch — it is a recorded preference, not yet an enforced one.
  Flagged so a future reader does not assume setting it changes summation.

## Alternatives considered

### A. Satellite dispatch crate (like `cjc-ad` / `cjc-quantum`)

Put the policy in a new `cjc-runtime-policy` crate routed from both
executors. **Rejected** — the satellite pattern exists to break dependency
cycles when dispatch needs a crate *above* `cjc-runtime`. The policy needs
nothing above `cjc-runtime` (it reads rayon, which `cjc-runtime` already
owns), so it lives directly in `cjc-runtime` next to `profile.rs`. Simpler,
no new crate.

### B. Wall-clock-based energy

Estimate joules from elapsed time × package power. **Rejected** — wall time
is non-deterministic, so the estimate would vary run-to-run and violate
Prime Directive #3. Workload-count estimation is deterministic.

### C. Per-call rayon `pool.install()`

Instead of configuring the global pool once, wrap every parallel kernel in a
scoped pool. **Rejected for Phase 1** — invasive (touches every
`par_chunks_mut` site) for no determinism benefit; the set-once global pool
matches the "bounded execution fixed at process start" model.

### D. Advisory-only thread cap (don't touch rayon)

Store and report the cap but never enforce it. **Rejected** — toothless;
`--threads`/`--profile` would not actually reduce CPU saturation, defeating
the thermal goal.

## Tests

| Flavor | Location | Count |
|---|---|---|
| Unit | `crates/cjc-runtime/src/runtime_policy.rs` (`#[cfg(test)]`) | 16 |
| Wiring (AST↔MIR parity) | `tests/runtime_policy/wiring.rs` | 27 |
| Property (proptest, 256 cases each) | `tests/runtime_policy/proptest.rs` | 8 |
| Fuzz (bolero) | `tests/runtime_policy/fuzz.rs` | 3 |

Unit tests pin the presets, energy monotonicity/additivity, `effective_threads`
bounds, and enum round-trips. Wiring tests prove every builtin is byte-equal
across both executors (machine-dependent ones checked for parity only). Proptests
assert energy is non-negative/finite/monotone/additive and that mode setters
round-trip and reject unknown spellings. Fuzz harnesses confirm the dispatch
surface survives arbitrary op sequences, arbitrary mode strings (Err, never
panic), and arbitrary energy inputs (finite, non-negative).

## Source

- Branch: `claude/eloquent-lederberg-dad128`
- Crates touched: `cjc-runtime` (new `runtime_policy` module + builtins),
  `cjc-cli` (flags + startup application)
- Concept note: [[Runtime Policy Layer]]

## Related

- [[Runtime Policy Layer]] — the concept note + green-compute rationale
- [[ADR-0011 Parallel Matmul]] — why thread count doesn't change results
- [[Binned Accumulator]] / [[Kahan Summation]] — the fixed-order reductions
- [[Float Reassociation Policy]] — the no-FMA discipline the energy model follows
- [[Wiring Pattern]] — one dispatch arm, both executors
- [[Determinism Contract]] — the invariants this layer preserves
- [[TidyView Architecture]] / [[ADR-0023 ABNG Adaptive Belief Radix Graph (Phase 0.1)]] — beneficiaries of the deferred adaptive-layout + det-hash-table lever
