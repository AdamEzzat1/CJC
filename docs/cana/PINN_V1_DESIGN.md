# PINN v1 — Deterministic Physical-Cost Model (Design Note)

**Date:** 2026-06-10
**Branch:** `claude/objective-davinci-62975a`
**Prereq:** v2 baseline committed (`e0e819b` cana-compress + `66b65bd` SVD fix)
**Source plan:** `docs/cana_compress/PHASE_A_HANDOFF_V2.md` §4 (Track 1)

This is the STEP-2 design note (per CLAUDE.md workflow) for the
physics-informed layer v1: a **deterministic coefficient-based
physical-cost model** — no neural net, no trained weights, no shadow
mode. v2 (MLP via `cjc-ad::GradGraph`) is explicitly deferred and will
be a drop-in replacement behind the same trait surface.

---

## 1. What gets built, where

| File | Contents | New deps |
|---|---|---|
| `crates/cjc-cana/src/physical_cost.rs` | `PhysicalCostQuery`, `PhysicalCostEstimate`, `PhysicalCoefficients`, `PhysicalConstraints`, `predict_physical()`, `build_physical_query()` | none |
| `crates/cjc-cana/src/pinn_cost_model.rs` | `PinnPhysicalCostModel<M, P>` + `CostModel` impl | none |
| `crates/cjc-cana/src/quantum_score.rs` | docs-only module: three-role framing + combined scoring formula | none |
| `crates/cjc-cana-compress/src/energy.rs` | `EnergyComponents` 9 → 11 fields (see §4 — the handoff said 12; its arithmetic double-counted a rename) | none |
| `crates/cjc-cana-compress/src/energy_pass_ranker.rs` | `derive_energy_components` populates the new fields | none |
| `crates/cjc-cana-nss/src/pinn_bridge.rs` | `PhysicalCostEstimate` → NSS `PressureKind` mapping | none |
| `docs/cana/PHYSICS_INFORMED_CANA_NSS.md` | user-facing documentation | — |

Placement follows the handoff §8 Q1 decision: **modules inside
`cjc-cana`** for v1 (zero new deps — `cjc-cana` does NOT gain a
`cjc-nss` dependency; the `PressureKind` mapping lives in
`cjc-cana-nss::pinn_bridge` which already depends on both).

## 2. Deriving `PhysicalCostQuery` from integer features

`FnFeatures` is integer-only (`CfgMetrics` + `MemoryProxy`, all `u32`).
`build_physical_query()` is a pure saturating-integer function of those
counters plus the queried pass name:

```text
loop_amp        = 1 + 7 * min(max_loop_depth, 4)        // 1, 8, 15, 22, 29
flops_estimate  = expr_count as u64 * loop_amp
bytes_read      = (tensor_heavy_ops * 256 + expr_count * 8) * loop_amp
bytes_written   = (cow_write_sites * 256 + alloc_sites * 64) * loop_amp
allocation_bytes= alloc_sites as u64 * 64
working_set     = alloc_sites * 64 + tensor_heavy_ops * 1024
thread_count    = 1                                      // CJC is single-thread deterministic today
batch_size      = 1
```

All multiplications saturating. The byte scales (64 / 256 / 1024) are
documented proxies, not measurements — Phase A6's profile DB exists to
replace them with trained values in v2. The pass being queried
amplifies the estimates via a per-pass factor table (mirroring
`pass_code_size_factor` in the energy ranker):

```text
pass_physical_amp("loop_unroll")  = (flops x2, code_size x2)
pass_physical_amp("vectorize")    = (flops x1, bytes x2)
pass_physical_amp("specialize")   = (code_size x2)
pass_physical_amp("monomorphize") = (code_size x3)
everything else                   = identity
```

## 3. `predict_physical()` — exact v1 formulas

Refining the handoff's placeholder formulas. Every output is a
deterministic, finite, clamped function of the integer query:

```text
norm(x, scale)     = x as f64 / (x as f64 + scale)        // smooth [0,1), monotone
heat_accumulation  = norm(flops, 1e7)
                   * (1 + thread_amplification * (threads-1))
                   * (1 + batch_amplification  * (batch-1))
thermal_pressure   = clamp01(heat_accumulation * (1 - cooling_rate))

bytes_moved        = bytes_read + bytes_written            (saturating)
memory_pressure    = clamp01(norm(allocation_bytes, 1e6)
                   + bytes_per_flop_scale * norm(bytes_moved, 1e8))
bandwidth_pressure = clamp01(norm(bytes_moved, 1e8))
alloc_churn        = clamp01(norm(allocation_bytes, 1e6) * alloc_churn_weight)
energy_estimate    = heat_accumulation + bandwidth_pressure + alloc_churn   // NOT clamped; additive Joule-proxy
locality_risk      = clamp01(locality_weight * norm(working_set, CACHE_PROXY=2^21))

confidence         = clamp01(1.0 - 0.5 * alloc_churn
                                  - thread_amplification * (threads-1))
```

Design choices vs the handoff sketch:
- `norm(x, scale)` (a rational sigmoid) instead of raw linear scaling —
  guarantees [0,1) without magic clamps for plausible inputs, stays
  monotone, has no overflow risk after the `as f64` conversion.
- No FMA anywhere; each term is bound to a named intermediate
  (`let t = a * b; t + c` style) per determinism invariant #3.
- `f64` ops only on values already reduced from integers — no
  accumulation loops, so no Kahan needed inside `predict_physical`.
- Coefficients default per the handoff: `heat_per_flop` is absorbed
  into the `1e7` flops normalization scale (a coefficient field
  `flops_norm_scale` keeps it tunable).

`PhysicalCoefficients::default()` keeps the handoff's eight knobs
(renaming `heat_per_flop` → `flops_norm_scale`, `bandwidth_per_byte` →
`bytes_norm_scale`, `alloc_churn_per_byte` → `alloc_norm_scale` to
match the normalized formulation; the doc-comment records the
correspondence).

## 4. `EnergyComponents`: 9 → 11 fields (not 12)

The handoff's §4.5 says "grows to 12" but lists exactly three changes,
one of which is a **rename**: `code_size_cost` → `code_size_pressure`.
9 + 2 genuinely-new + 1 rename = **11**. (The handoff's combined
formula also names `compile_cost`, which has never existed in the
struct — `runtime_cost` covers it. Not adding a dead field.)

Changes:
1. Rename field `code_size_cost` → `code_size_pressure`. Builder keeps
   `.code_size_cost()` as a `#[deprecated]` alias for one release.
2. New cost field `bandwidth_pressure: f64` (declared after
   `thermal_pressure`).
3. New reward field `locality_reward: f64` (declared after
   `compression_reward`).
4. `new()` grows to 11 args (single production caller per handoff —
   the energy ranker — plus tests).
5. `is_valid()`, `kahan_total()`, builder, proptest strategies updated.

**Kahan-order note:** `kahan_total()` sums in declaration order;
inserting `bandwidth_pressure` mid-sequence inserts a `0.0` add into
the compensation stream, which can in principle alter bit patterns.
No test asserts golden energy totals (determinism tests are double-run
identity), so this is safe — verified by running the full
cana-compress suite. If a golden test is ever added, it must be
re-locked at that point, not here.

## 5. `PinnPhysicalCostModel<M, P>` — composition semantics

Follows `ThermalAwareCostModel`'s interception pattern exactly:

- **Only `PassBenefit` queries** are adjusted. `PassRuntime` (a
  compile-cost) and `PeakMemory` pass through to the base.
- For a `PassBenefit{function_name, pass_name}` query:
  1. `base = base_model.query(...)` — if `Unknown`, return `Unknown`
     (never invent an estimate the base didn't make).
  2. `q = build_physical_query(features[function_name], pass_name)`.
  3. `est = predict_physical(q)`.
  4. **Hard rejection:** if any of `est.{thermal,memory,bandwidth}_pressure`
     exceeds its `PhysicalConstraints::max_*`, or
     `est.energy_estimate > max_energy_estimate` (when `Some`),
     return `Estimated { value: 0.0, confidence: 1.0 }` — a confident
     zero-benefit prediction that falls below the ranker's skip
     threshold (0.005) and is dropped as `BelowSkipThreshold` with the
     drop recorded. **Implementation note (discovered during wiring):**
     the handoff prescribed `CostEstimate::Unknown` here, claiming "the
     base ranker treats this as do-nothing" — that's wrong for this
     codebase. `PassRanker` maps `Unknown` to
     `UnknownButKeptConservatively` (the pass is KEPT), which would
     make hard limits a no-op. Confident-zero is the mechanism that
     actually withholds a recommendation. The legality gate is NOT
     involved; PINN can only *withhold* recommendations, never approve.
  5. **Soft blend:** `value' = max(0, value - physical_penalty)` where
     `physical_penalty = prefer_cooler_plan_margin *
     (thermal + memory + bandwidth pressures)`. Scaling by the margin
     (default 0.1) keeps v1 bias-ordering rather than dominating the
     trained linear coefficients.
  6. `confidence' = clamp01(confidence * est.confidence)`.
- `name() = "pinn_coeffs_v1"`, `version() = 1` — flows into
  `CanaReport` hashing exactly like every other model (invariant #8).

**Double-counting (handoff Q8):** `PinnPhysicalCostModel` *replaces*
`ThermalAwareCostModel` in the recommended stack:

```text
recommended:  LinearCostModel → PinnPhysicalCostModel
not:          LinearCostModel → ThermalAwareCostModel → PinnPhysicalCostModel
```

Both wrappers penalize thermal on `PassBenefit`; stacking them applies
thermal influence twice. This is documented on the struct + enforced
socially (a debug-build warning is overkill for an advisory layer; a
unit test demonstrates the recommended stack's output differs from the
double-stacked one, locking the documentation claim).

The `P: PressurePredictor` parameter lets v1 consult NSS-predicted
thermal as a *blend input* alongside the closed-form heat term:
`thermal_pressure = max(formula_thermal, nss_thermal[function])` —
taking the max is conservative, deterministic, and keeps the NSS
signal advisory.

## 6. `pinn_bridge` (cjc-cana-nss)

Pure function surface, no state:

```rust
pub fn physical_estimate_to_pressure_deltas(est: &PhysicalCostEstimate)
    -> Vec<(PressureKind, f64)>   // sorted by PressureKind discriminant, clamped [0,1]
```

Mapping per handoff §4.4: thermal→`Thermal`, memory→`Memory`,
bandwidth→`Throughput` (+`Io` at half weight when bytes-heavy),
energy→`Cpu` (half weight composite), locality→`Memory` (quarter
weight, working-set saturation). `confidence` is not a pressure and is
not emitted. Tests prove determinism + clamping + stable ordering.

## 7. Test plan

- `physical_cost.rs` unit: finiteness + clamping over a coefficient
  grid; zero query → zero pressures; monotonicity (more flops → ≥
  thermal); saturating arithmetic at `u64::MAX`; default coefficients
  sensible on a synthetic workload.
- `pinn_cost_model.rs` unit: NullCostModel composition (Unknown stays
  Unknown); ConstantCostModel + hard limits reject; soft blend demotes
  hotter strategies; PassRuntime/PeakMemory pass-through; determinism
  ×50; recommended-stack vs double-stack difference test.
- `energy.rs`: existing 155-suite must stay green; builder alias test;
  11-field validity; proptest strategies extended.
- `pinn_bridge`: determinism, clamping, ordering, all-zero estimate →
  all-zero deltas.
- Bolero fuzz (`cjc-cana-compress/tests/bolero_fuzz.rs` extension or
  new target in cjc-cana): random physical queries (struct-filling
  from arbitrary bytes) must never panic and always produce finite
  clamped outputs; random coefficient sets likewise.
- The load-bearing gate: `cargo test --test fixtures --release`
  (PINN is compile-advisory only — MIR output unchanged unless the
  ranker changes plans, and the legality gate still bounds that).

## 8. Explicitly deferred (unchanged from handoff)

Phase A4 (profile DB), A5 (compression delta into NssPressurePredictor),
A6 (training-record emission), and all of v2 (MLP, residual losses,
shadow-mode promotion). The `compression_overhead` field on
`PhysicalCostQuery` ships as an `Option<u64>`-style passive field
populated by the bridge extension (§4.6) only if time allows this
session; otherwise it lands with A5.
