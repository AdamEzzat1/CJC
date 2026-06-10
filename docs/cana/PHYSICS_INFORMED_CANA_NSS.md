# Physics-Informed CANA/NSS — v1 (Deterministic Coefficient Model)

**Date:** 2026-06-10
**Status:** Shipped (v1). v2 (neural-trained) deferred until ≥1000 profile rows.
**Design note:** [PINN_V1_DESIGN.md](PINN_V1_DESIGN.md)
**Phase-A context:** [../cana_compress/PHASE_A_HANDOFF_V2.md](../cana_compress/PHASE_A_HANDOFF_V2.md)

## What this layer is

A deterministic physical-cost model that prices compiler strategies in
physical terms — heat, memory traffic, bandwidth, allocation churn,
cache locality — and feeds those prices into CANA's pass-ranking
decisions and NSS's pressure substrate.

v1 contains **no neural network and no trained weights**. It is a
closed-form function of integer workload estimates with hand-tuned
coefficients. That makes it deterministic by construction: no shadow
mode, no training data, no model-drift concerns. v2 will swap the
coefficient table for a small MLP behind the *same* trait surface.

## The four roles of the advisory stack

The quantum-inspired layer shipped in Phase 6 covers three roles;
this layer adds a fourth:

| Role | Primitive | Where |
|---|---|---|
| Compression | tensor-train / motif / RLE advisory compression | `cjc-cana-compress/src/{lossless_trace,motif_dictionary,lowrank,tensor_train}.rs` |
| Search / ranking | Ising-style `EnergyRanker` + `EnergyAwarePassRanker` | `cjc-cana-compress/src/{energy,energy_pass_ranker}.rs` |
| State representation | `PressureDensityState` (density-matrix-shaped) | `cjc-nss/src/density.rs` |
| **Physical cost** (NEW) | `PinnPhysicalCostModel` + `physical_cost` | `cjc-cana/src/{physical_cost,pinn_cost_model}.rs` |

None of these replaces another; they compose through traits.

## Architecture

```text
MirProgram + CanaFeatures (integer MIR-shape features)
   │
   ├─► build_physical_query(fn, pass, features)     [cjc-cana::physical_cost]
   │        integer workload estimates: flops, bytes r/w, alloc, working set
   │        (saturating arithmetic; loop-depth + per-pass amplification)
   │
   ├─► predict_physical(query, coefficients)        [cjc-cana::physical_cost]
   │        → PhysicalCostEstimate { thermal, memory, bandwidth,
   │                                 energy, locality, confidence }
   │        every pressure finite + clamped [0,1]
   │
   ├─► PinnPhysicalCostModel<M, P>                  [cjc-cana::pinn_cost_model]
   │        wraps base CostModel M + PressurePredictor P
   │        • PassBenefit queries: hard-reject or soft-blend
   │        • thermal = max(closed-form, NSS-predicted)   (conservative)
   │        • PassRuntime / PeakMemory: pass through
   │
   ├─► PassRanker → EnergyAwarePassRanker           [existing seams]
   │        EnergyComponents grew 9 → 11 fields:
   │        + bandwidth_pressure (cost), + locality_reward (reward),
   │        code_size_cost renamed → code_size_pressure
   │
   └─► physical_estimate_to_pressure_deltas(est)    [cjc-cana-nss::pinn_bridge]
            → [(Cpu, δ), (Memory, δ), (Io, δ), (Thermal, δ), (Throughput, δ)]
            for NSS pressure-field consumers

LegalityGate + MIR verifier retain final authority. PINN is advisory.
```

## Hard rejection vs soft blend

`PhysicalConstraints` (defaults: `max_*_pressure = 0.95`,
`prefer_cooler_plan_margin = 0.1`, `max_energy_estimate = None`):

- **Hard:** an estimate exceeding any `max_*` produces
  `Estimated { value: 0.0, confidence: 1.0 }` — a confident
  zero-benefit prediction. `PassRanker` drops it below the skip
  threshold (`BelowSkipThreshold`) and records the drop.
  ⚠ NOT `CostEstimate::Unknown`: this codebase's ranker KEEPS
  unknown-benefit passes (`UnknownButKeptConservatively`). The
  Phase-A handoff's assumption that `Unknown` means "do nothing" was
  wrong; the wiring test caught it.
- **Soft:** surviving strategies get
  `value −= margin × (thermal + memory + bandwidth)`, floored at 0,
  and `confidence ×= physical_confidence`. With the default 0.1
  margin, v1 bias-orders rather than dominating the trained linear
  coefficients.

## Stack placement

```text
recommended:  LinearCostModel → PinnPhysicalCostModel
NOT:          LinearCostModel → ThermalAwareCostModel → PinnPhysicalCostModel
```

`PinnPhysicalCostModel` supersedes `ThermalAwareCostModel` (single-axis
thermal) with a multi-axis physical model. Stacking both applies
thermal influence twice — the `stacking_double_penalizes_thermal` unit
test demonstrates the distortion.

## Determinism

- `predict_physical` is a pure function of `(integer query, coefficients)`;
  the integer side uses saturating arithmetic, the float side a fixed
  sequence of named intermediates (no FMA contraction, no loops, no
  Kahan needed).
- `model_id = "pinn_coeffs_v1"`, `model_version = 1` flow into report
  hashing via `CostModel::name()/version()` (invariant #8).
- Invalid coefficients (NaN scales, `cooling_rate ≥ 1`) make the layer
  **abstain** (base estimate passes through) — never panic, never
  reject.
- Bolero fuzz: arbitrary bit-pattern queries + coefficient sets can
  only produce valid clamped estimates or provable abstention.

## NSS PressureKind mapping (pinn_bridge)

| Estimate field | PressureKind | weight |
|---|---|---|
| `thermal_pressure` | `Thermal` | 1.0 |
| `memory_pressure` | `Memory` | 1.0 |
| `locality_risk` | `Memory` | 0.25 (working-set echo) |
| `bandwidth_pressure` | `Throughput` | 1.0 |
| `bandwidth_pressure` | `Io` | 0.5 (v1 can't split disk traffic) |
| `energy_estimate` (clamped) | `Cpu` | 0.5 |
| `confidence` | — | not a pressure; not emitted |

## Test inventory (this layer)

| Suite | Tests |
|---|---|
| `cjc-cana::physical_cost` unit | 13 |
| `cjc-cana::pinn_cost_model` unit | 11 |
| `cjc-cana-nss::pinn_bridge` unit | 7 |
| `cjc-cana-compress` energy 11-field updates (new/changed) | +2 in-module |
| `tests/pinn_wiring.rs` end-to-end | 4 |
| `tests/bolero_fuzz.rs` PINN targets | 2 (×1000 iterations each) |
| AST/MIR parity gate | passes (PINN is compile-advisory only) |

## Deferred to later sessions

- Phase A4 (expected-vs-actual profile DB), A5 (compression delta into
  `NssPressurePredictor`), A6 (training-record emission) — the
  training corpus prerequisites for v2.
- `PhysicalCostQuery::compression_overhead` (handoff §4.6) — lands
  with A5.
- v2 neural training: MLP via `cjc-ad::GradGraph`, physics-residual
  soft losses, shadow-mode → active-mode promotion gate (handoff §5.4).
