# ABNG Phase 0.3c — OOD + Calibration + Drift (Design Note)

**Date:** 2026-05-06
**Builds on:** [Phase 0.3b](PHASE_0_3b_DESIGN.md)
**Scope:** Three independent per-node uncertainty subsystems — density tracker, calibration bins, drift baseline — plus a composite OOD score that combines them with the BLR head's epistemic signal. **No structural decisions** — those arrive in 0.3d.

## Why this slice

Phase 0.3b made ABNG defensibly Bayesian-inspired (BLR for last-layer
uncertainty). Phase 0.3c gives it the *additional* signals a calibrated
Bayesian-inspired model is supposed to expose:

* **OOD score** — is this query coming from the training distribution at all?
* **Calibration error** — when the model says it's 80% sure, is it right 80% of the time?
* **Drift score** — has the inference traffic shifted away from training?

These three are independent in 0.3c. Composing them into a single
abstain/predict/escalate decision is Phase 0.3d's structural-decision
engine — the policy lives there, the *evidence* lives here.

What 0.3c *does* deliver:

1. `DensityTracker` per node (Welford μ + diagonal M2 over BLR-feature
   space). Mahalanobis distance + density score `1 − exp(−mahal²)`.
2. `CalibrationBins` per node (15-bin reliability diagram; ECE).
3. `DriftBaseline` per node — a snapshot of the density tracker at a
   user-triggered freeze. `drift_score = ‖μ_current − μ_baseline‖` rescaled
   by baseline std.
4. Composite `ood_score(node, phi, matched_prefix) → f64` =
   `max(density_score, prefix_distance, epistemic_z)`.
5. Snapshot format **v5** (clean break from v4).
6. ~11 new builtins (512 → 523).

What 0.3c does **not** deliver:

* Structural decisions (Grow/Split/Merge/Prune/Compress/Freeze) — Phase 0.3d.
* Full covariance Mahalanobis — diagonal is enough for OOD detection;
  full is a 0.3d optimisation if specific failure modes show up.
* Categorical-shift / route-shift drift signals — those need the
  multi-output / multinomial heads that don't exist yet. 0.3c's drift
  is feature-mean shift only.
* `Maturity` + `Freeze` policy plumbing — 0.4 alongside the CLI.

## Three independent modules

### `density.rs` — `DensityTracker`

```rust
pub struct DensityTracker {
    pub d:    u32,
    pub n:    u64,
    pub mean: Tensor,    // [d] — Welford running mean
    pub m2:   Tensor,    // [d] — diagonal Welford M2 (variance × (n-1))
}
```

Welford update on a feature vector `phi: [d]`:

```text
n     ← n + 1
delta ← phi − mean
mean  ← mean + delta / n
delta2← phi − mean
m2    ← m2 + delta · delta2     (elementwise)
```

Variance: `m2 / (n − 1)` per dim, with `0.0` floor for `n < 2`.

**Mahalanobis** (diagonal):
```text
mahal² = Σᵢ (phiᵢ − meanᵢ)² / max(varianceᵢ, ε)
```

`ε = 1e-12` keeps the divide bounded when a feature is constant. The
score is bounded:

```text
density_score = 1 − exp(−mahal²)        ∈ [0, 1)
```

Returns `0.0` when `n < 2` (no signal yet), forcing Phase 0.3d-policy
to abstain on density alone.

### `calibration.rs` — `CalibrationBins`

```rust
pub struct CalibrationBins {
    pub n_bins:        u8,             // 15 by default
    pub counts:        Vec<u32>,       // [n_bins]
    pub correct_counts:Vec<u32>,       // [n_bins]
    pub conf_sum_bits: Vec<u64>,       // [n_bins], Kahan-finalized f64::to_bits
}
```

`observe(predicted_prob ∈ [0, 1], was_correct: bool)`:
1. `bin = floor(predicted_prob × n_bins)`, clamped to `n_bins − 1`
2. `counts[bin] += 1; if was_correct { correct_counts[bin] += 1 }`
3. Kahan-add `predicted_prob` to `conf_sum[bin]` (stored as bit pattern).

ECE:
```text
N   = Σ counts
ECE = Σ (counts[b] / N) · |correct_counts[b]/counts[b] − conf_sum[b]/counts[b]|
```
where empty bins contribute zero.

The conf_sum is stored as bit pattern of the Kahan-finalized f64 to
keep canonical-bytes encoding deterministic across runs without leaking
the compensation register's mid-state into the snapshot.

### `drift.rs` — `DriftBaseline`

```rust
pub struct DriftBaseline {
    pub d:        u32,
    pub mean:     Tensor,    // [d] — frozen baseline mean
    pub std:      Tensor,    // [d] — frozen baseline std
    pub n_at_freeze: u64,
    pub frozen_hash: [u8; 32],
}
```

Frozen by an explicit `freeze_drift_baseline(node)` call (per-node, can
fire any time after the density tracker has at least 2 observations).
`drift_score(current_density, baseline)`:

```text
shift = Σᵢ ( (μ_current[i] − μ_baseline[i]) / max(std_baseline[i], ε) )²
score = sqrt(shift / d)        ∈ [0, ∞)
```

This is a per-dim z-shift L2-normalized. Larger = more drift. No
threshold is baked in — user code sets the cutoff.

## Composite OOD score

```rust
pub fn ood_score(
    density:        Option<&DensityTracker>,
    blr:            Option<&BlrState>,
    phi:            &[f64],
    matched_prefix: u8,
    prefix_max:     u8,
) -> f64
```

```text
density_score    = density.map(d| d.density_score(phi)).unwrap_or(0.0)
prefix_distance  = if prefix_max > 0 { (prefix_max − matched_prefix) / prefix_max } else { 0.0 }
epistemic_z      = blr.map(b| (b.epistemic_var(phi) / b.expected_epistemic).clamp(0, 1)).unwrap_or(0.0)
OOD              = max(density_score, prefix_distance, epistemic_z)
```

`max` (not `mean`) — any *single* strong signal triggers the highest
OOD reading, so abstain logic is conservative by default.

## New audit kinds (5)

| Tag  | Kind | Payload |
|------|------|---------|
| 0x0B | `DensityTrackerInstalled` | `state_hash: [u8; 32]` (per-node) |
| 0x0C | `DensityUpdated` | `state_hash: [u8; 32]` (per-node, post-update) |
| 0x0D | `CalibrationInstalled` | `state_hash: [u8; 32]` |
| 0x0E | `CalibrationUpdated` | `state_hash: [u8; 32]` |
| 0x0F | `DriftBaselineFrozen` | `state_hash: [u8; 32]` |

All carry the 32-byte witness only; the actual subsystem state lives in
the per-node section of the snapshot. Same trade-off Phase 0.3a/b
established.

## Snapshot format v5

Header gains `density_present u8` + `calibration_present u8` flags right
after the BLR-prior section. If `density_present`, the next byte is
`d_dim` (the BLR-feature dim); if `calibration_present`, the next is
`n_bins u8`. Both are graph-wide installation choices.

Per-node section gains three new optional blobs, each prefixed with a
`u8` presence flag:

* `density: Option<DensityTracker>` — 1 + 32 bytes (d, n) + d × 16 bytes (mean, m2) when present
* `calibration: Option<CalibrationBins>` — 1 + 1 byte (n_bins) + 16n_bins bytes (counts, correct_counts, conf_sum_bits) when present
* `drift_baseline: Option<DriftBaseline>` — 1 + 4 (d) + d × 16 (mean, std) + 8 (n_at_freeze) + 32 (frozen_hash) when present

Replay verifies each blob's hash against the most-recent
`Installed`/`Updated`/`Frozen` event for that node, exactly like the
Phase 0.3a/b params + BLR pattern.

## New builtins (11)

| Name | Args | Returns | Purpose |
|---|---|---|---|
| `abng_set_density_tracker` | `g` | `Void` | install + freeze; per-leaf init at d=BLR-feature dim |
| `abng_density_observe` | `g, node_id, features_2d` | `Void` | Welford update from a batch |
| `abng_density_score` | `g, node_id, phi_1d` | `Float` | density score `1 − exp(−mahal²)` |
| `abng_density_n_seen` | `g, node_id` | `Int` | observations applied |
| `abng_set_calibration` | `g, n_bins: i64` | `Void` | install + freeze (n_bins ∈ [2, 100]) |
| `abng_calibration_observe` | `g, node_id, predicted_prob: f64, was_correct: bool` | `Void` | bin update |
| `abng_calibration_ece` | `g, node_id` | `Float` | weighted-gap ECE |
| `abng_calibration_n_seen` | `g, node_id` | `Int` | total bin entries |
| `abng_freeze_drift_baseline` | `g, node_id` | `Void` | snapshot current density as baseline |
| `abng_drift_score` | `g, node_id` | `Float` | z-shift L2 of current vs baseline |
| `abng_ood_score` | `g, node_id, phi_1d, matched_prefix: i64, prefix_max: i64` | `Float` | composite |

Total surface after 0.3c: **523 dispatch arms** (512 + 11).

## Tests (estimate)

* In-crate: 94 → ~115 (+21) — Welford determinism, Mahalanobis edge cases, ECE
  on synthetic reliability diagrams, drift baseline freeze + score, snapshot
  v5 round-trip with all three subsystems.
* Integration: 148 → ~180 (+32) — every new builtin, AST↔MIR parity, end-to-end
  "train BLR + track density + freeze drift baseline + observe shift → drift_score
  rises" round-trip.

## Risks

1. **Diagonal covariance assumption.** Off-axis correlation in features
   isn't captured. Mitigation: documented; full covariance is a 0.3d
   opt; in practice for chess-RL value heads (post-tanh penultimate
   features) the covariance is approximately diagonal anyway.
2. **ECE on small bins.** Bins with <5 samples are noisy. Mitigation:
   Phase 0.3d's `Maturity.calibration_stable` will require minimum
   bin populations before declaring stable. Phase 0.3c just exposes
   the raw ECE — let policy decide.
3. **Drift baseline freeze timing.** When the user calls
   `freeze_drift_baseline()` matters. Mitigation: it's user-policy;
   ABNG only provides the primitive. A reasonable default in 0.3d:
   freeze when `Maturity.uncertainty_stable` first holds.
4. **Snapshot v4 → v5 break.** Same justification as v3 → v4.
