---
title: "Cronos GAN Phase 4d → 6 Handoff"
tags: [handoff, planning, roadmap, cronos-gan, temporal-gan, predictor-challenger, holdout-eval]
status: 📋 Planned — three workstreams queued after Phase 4c's empirical flip
crate: cjc-cronos-gan
date: 2026-06-04
date-modified: 2026-06-04
---

# Cronos GAN Phase 4d → 6 Handoff

Follows Phase 4c (`ce2f0d3` on `feat/cjc-cronos-gan`) where the held-out evaluation flipped the Phase 4b conclusion: `liquid_as_generator`'s training-MSE advantage was overfitting; `ssm_as_generator` produces the only calibrated disagreement that transfers to held-out data. This handoff covers the remaining Phase 4 work, all of Phase 5, and Phase 6, organized so each phase ships an independently-defensible artifact.

## 0. TL;DR

| Phase | Headline deliverable | LOC est. | Tests est. | Sessions |
|---|---|---|---|---|
| **4d** | λ schedule (Constant / Linear / ExponentialDecay) + per-mode `lr` + multi-seed mean+variance per cell | ~500 | ~25 | 1.5 |
| **5b** | 7 Bolero fuzz targets + `tests/cronos/{unit,integration,prop,fuzz}/` workspace layout + cross-platform CI matrix activation | ~400 | 7 fuzz + 5 wiring | 1 |
| **6** | `cjc-locke` E9500+ custom detector consuming `eval.disagreement.regime_shift_score` + vault deep-docs (6 files) + Python bridge via maturin | ~600 + 6 doc files + 1 binding crate | ~20 + Python pytest | 2-3 |

**Critical interdependency**: Phase 5b's cross-platform CI matrix should land BEFORE Phase 6's Locke detector. The detector emits content-addressed findings; if the cross-platform byte-identity claim isn't validated, downstream Locke consumers can't trust the findings' replay properties.

**Branch state**: `feat/cjc-cronos-gan` at `ce2f0d3`, local-only, 131 tests passing release.

## 1. Phase 4d — λ schedule + per-mode lr + multi-seed sweep (~1.5 sessions)

### 1.1 What ships

The Phase 4c empirical flip was on **one seed**. The Phase 4b → 4c reinterpretation hinges on `liquid_as_generator`'s overfitting being a consistent phenomenon, not a single-seed artifact. Phase 4d adds the statistical machinery to make that distinction empirically tight, plus the λ-schedule the Phase 4c open question called out.

**Code additions (~500 LOC):**

```rust
// New enum, defaults to Constant for backward compat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LambdaSchedule {
    /// λ is the same on every training step.
    Constant,
    /// λ decays linearly from `lambda_start` at step 0 to `lambda_end`
    /// at step `n_train_steps`.
    Linear { start: f64, end: f64 },
    /// λ_t = lambda_start · exp(-decay_rate · t / n_train_steps).
    ExponentialDecay { start: f64, decay_rate: f64 },
    /// λ_t = lambda_start for t < warmup_steps; then linear decay to lambda_end.
    WarmupThenLinear { start: f64, end: f64, warmup_steps: usize },
}

// TemporalGanConfig.lambda_disagreement: f64 → lambda_schedule: LambdaSchedule
//   with a convenience constructor that wraps a scalar into Constant.
// TemporalGanTrainer caches `schedule.lambda_at(step_count)` per step and
// passes it into ChallengerSpec.

// SweepBaseConfig gains:
pub per_mode_lr: BTreeMap<TemporalGanMode, f64>,
pub n_seeds: usize,  // default 1 (backward compat with Phase 4c)
pub seed_stride: u64, // SplitMix step constant; PER_CHAIN_MIX from metropolis

// New per-mode lr accessor `lr_for(mode)` falls back to default_lr (mirrors
// lambda_for pattern from Phase 4c).

// run_experiment_sweep iterates n_seeds: for each cell, runs the experiment
// with seeds [base_seed, base_seed + stride, base_seed + 2·stride, ...].
// Returns ExperimentSweepReport with per-cell mean + variance + per-seed
// replay_hashes vec.
```

**ExperimentSweepReport extensions:**

```rust
pub struct SweepCell {
    pub dataset: CronosDataset,
    pub mode: TemporalGanMode,
    pub per_seed_reports: Vec<ExperimentReport>,   // length = n_seeds
    pub mean: CellAggregate,                        // Kahan-summed across seeds
    pub variance: CellAggregate,                    // Welford
}

pub struct CellAggregate {
    pub final_loss_ssm: f64,
    pub final_loss_liquid: f64,
    pub mean_absolute_gap: f64,
    pub max_regime_shift_score: f64,
    pub eval_ssm_loss: Option<f64>,
    pub eval_liquid_loss: Option<f64>,
    pub eval_absolute_gap: Option<f64>,
}
```

Mean and variance via `cjc_repro::KahanAccumulatorF64` + Welford recurrence. `sweep_hash` salts with the per-seed replay_hashes so the determinism contract scales naturally.

**`format_table` extensions:**

Two flavors of output: per-cell-per-seed (the current 15-row table × `n_seeds`) and aggregate (15 rows showing `mean ± std` for each metric). Default to aggregate when `n_seeds > 1`; per-seed when `n_seeds == 1` (matches Phase 4c behavior).

### 1.2 Tests (~25)

| Group | Coverage |
|---|---|
| `LambdaSchedule` math (5) | Constant returns same value every step; Linear interpolates correctly; ExponentialDecay matches closed form; WarmupThenLinear stays constant during warmup; lambda_at boundaries (t=0, t=n_train_steps) |
| Schedule integration (3) | `with_lambda_schedule(Linear)` produces different params than `Constant` after equal steps; same schedule + same seed → byte-identical training trajectory; constant schedule + same scalar λ → byte-identical to Phase 4c |
| Per-mode `lr` (2) | Override changes only that mode's cells; `lr_for(mode)` fallback to `default_lr` |
| Multi-seed sweep (5) | `n_seeds = 3` produces 3 reports per cell; per-seed `replay_hash`es are distinct; mean across seeds is finite + matches manual Kahan-summed mean; variance is non-negative; aggregate `sweep_hash` byte-identical across runs |
| Multi-seed determinism (3) | Same `(SweepBaseConfig, base_seed, seed_stride, n_seeds)` → byte-identical `sweep_hash`; changing `seed_stride` shifts the hash; changing `n_seeds` shifts the hash |
| Format adaptation (3) | `n_seeds=1` produces per-cell rows; `n_seeds>1` produces `mean ± std` rows; doc test for the format example |
| **Phase 4c empirical flip replication (4)** | Across `n_seeds=10`, the inequalities `eval_ssm_loss(ssm_as_gen) < eval_ssm_loss(liq_as_gen)` and `eval_absolute_gap(ssm_as_gen) < eval_absolute_gap(symmetric)` and `eval_absolute_gap(ssm_as_gen) < eval_absolute_gap(liq_as_gen)` hold on the MEAN (use `n_seeds = 10`, generous tolerance). This is the test that turns Phase 4c's single-seed flip into a statistical claim. |

### 1.3 Open empirical question Phase 4d answers

**Does decaying λ in `liquid_as_generator` recover the training-MSE advantage while preserving generalization?** Run the canonical sweep at `LambdaSchedule::Linear { start: 0.15, end: 0.0 }` for `liquid_as_generator`. If the eval SSM MSE matches `symmetric` (regularization warmup, then converge to vanilla supervised) AND the training SSM MSE drops below symmetric (NCL diversification during the warmup phase), the predictor/challenger framing gains a third practical recipe. If not, the Phase 4c conclusion "pick SsmAsGenerator" hardens.

### 1.4 Risks / mitigations

- **`n_seeds = 10` × 15 cells × eval = 150 full experiments per sweep**: at ~0.4 s/experiment in release, that's ~60 s — still well within the CI matrix budget. Doable.
- **`LambdaSchedule` adds to canonical_bytes** — must hash *every* schedule variant's fields, not just the discriminant. Easy to get wrong; add a test that confirms two schedules with same discriminant + different fields produce different bytes.

## 2. Phase 5b — Bolero fuzz + workspace test layout + CI matrix activation (~1 session)

### 2.1 What ships

Phase 5 partial shipped proptest. Phase 5b adds adversarial-input fuzz coverage (the brief's original §7 list) plus the workspace-level test directory layout (`tests/cronos/{unit,integration,prop,fuzz}/`) plus activating the cross-platform CI matrix YAML committed in Phase 5 partial but not yet exercised on PR runs.

**Bolero targets (7, ~400 LOC):**

| Target | Generates | Asserts |
|---|---|---|
| `fuzz_malformed_temporal_batch` | Random `Vec<TimeSeries>` with mismatched `n_dim` | Constructor returns `DimensionMismatch`, no panic |
| `fuzz_random_sequence_lengths` | Random `n_steps` ∈ [0, 1024] | Either valid rollout or structured error; finite output if Ok |
| `fuzz_random_masks` | Random `SequenceMask` lengths vs series lengths | Either `MaskLengthMismatch` or successful construction |
| `fuzz_random_seeds` | Random `u64` seeds | Same seed twice → byte-identical params for both networks |
| `fuzz_random_ssm_configs` | Random `(state_dim, input_dim, output_dim, alpha, init_scale)` | Either valid model + `‖A‖_F ≤ alpha` or `InvalidConfig` |
| `fuzz_random_liquid_configs` | Random `(state_dim, input_dim, output_dim, dt, tau_min, tau_max, init_scale)` | Either valid model + `tau ∈ [tau_min, tau_max]` over a sample step or `InvalidConfig` |
| `fuzz_random_train_step_inputs` | Random `(inputs, targets, predictor_outputs)` for `TemporalGanTrainer.step` | No panic; finite loss; no NaN propagation without explicit `NonFiniteInput` error |

Each target uses `bolero::check!()` with `with_iterations(1024)` to hit ~10⁴ structurally-fuzzed cases per CI run.

**`tests/cronos/{unit,integration,prop,fuzz}/` layout (~5 wiring tests):**

The brief asked for this in Phase 5 but I declined at the time because crate-level `tests/` was cleaner. Phase 5b is the right moment because fuzz targets justify the directory split.

```
tests/cronos/
├── mod.rs               // [[test]] entry point in workspace Cargo.toml
├── unit/
│   ├── mod.rs
│   └── (re-export crate-level inline unit tests via #[path] modules)
├── integration/
│   ├── mod.rs
│   ├── training.rs      // moved from tests/test_training.rs
│   ├── gan_training.rs  // moved from tests/test_gan_training.rs
│   ├── sweep.rs         // moved from tests/test_experiment_sweep.rs
│   └── phase_4c.rs      // moved from tests/test_phase_4c.rs
├── prop/
│   ├── mod.rs
│   └── proptest_suite.rs // moved from tests/test_proptest.rs
└── fuzz/
    ├── mod.rs
    └── (7 Bolero target files)
```

Workspace `Cargo.toml` gets one new `[[test]]` block:
```toml
[[test]]
name = "cronos"
path = "tests/cronos/mod.rs"
```

This keeps the per-test-binary build cost down (Cargo builds one binary instead of one per file).

**CI matrix activation:**

Phase 5 partial committed `.github/workflows/cjc-cronos-gan-determinism.yml` but it's only triggered on `paths: crates/cjc-cronos-gan/**`. Once the branch pushes, the workflow fires on the first PR. Phase 5b expands the trigger to also include `tests/cronos/**` and the workflow YAML itself, and adds a `cronos-gan-multi-seed-byte-identity` job that runs `run_experiment_sweep` with `n_seeds=3` on all three platforms and asserts the resulting `sweep_hash`es are bit-identical.

### 2.2 Tests (5 wiring + 7 fuzz × 1024 iterations)

- 5 wiring tests for the `tests/cronos/{...}` layout that confirm each subdirectory's `mod.rs` correctly aggregates its files
- Each fuzz target's `bolero::check!()` is itself a test that runs the configured iterations on every CI run

### 2.3 Risks / mitigations

- **Bolero adds compile time** — first-build cost can spike. Mitigation: only fuzz targets compiled in `--release` for CI; inline tests stay fast in `--debug`.
- **The workspace test layout refactor moves files** — Phase 4d's tests need to be moved too. Either Phase 5b waits for 4d or it explicitly synchronizes the move.

## 3. Phase 6 — Locke detector + vault deep-docs + Python bridge (~2-3 sessions)

This is the "make Cronos GAN consumable" phase. Everything before this was about the internal architecture; Phase 6 is about exposing it to other parts of the workspace and to Python.

### 3.1 cjc-locke E9500+ custom detector

cjc-locke v0.8 ships a custom-detector extension layer (ADR-0041) reserving the `E9500..=E9999` code range for non-built-in detectors. Cronos GAN's regime-shift score is a natural fit.

**Code (~250 LOC in `crates/cjc-cronos-gan/src/locke_detector.rs`):**

```rust
use cjc_locke::custom_detector::{CustomDetector, FindingSink, BeliefAxisSet};
use cjc_locke::report::ValidationFinding;

/// Phase 6: Cronos GAN regime-shift detector for cjc-locke.
///
/// Consumes an `ExperimentReport` and emits Locke findings for
/// regime-shift signals exceeding the configured threshold.
pub struct CronosRegimeShiftDetector {
    pub regime_shift_threshold: f64,    // E9500 fires when eval.disagreement.regime_shift_score > this
    pub absolute_gap_threshold: f64,    // E9501 fires when eval.disagreement.absolute_gap > this
    pub ssm_loss_degradation_threshold: f64, // E9502 fires when eval_ssm_loss/train_ssm_loss > this
}

impl CustomDetector for CronosRegimeShiftDetector {
    fn run(
        &self,
        df: &cjc_locke::custom_detector::PyDetectorDataFrame, // or Rust adapter
        sink: &mut FindingSink,
    ) {
        // df carries the ExperimentReport as a serialized record column
        ...
    }
}
```

**E9500..=E9502 code definitions** (reserved range):

- `E9500` Cronos GAN regime-shift detected — `regime_shift_score > threshold`. Severity Warning. Evidence: per-step `regime_shift_score` trajectory + offending step indices. Suggests further investigation of the data window.
- `E9501` Persistent disagreement on held-out window — `eval.absolute_gap > threshold`. Severity Notice. Evidence: training vs eval `absolute_gap`. Suggests the data distribution shifted between train and eval.
- `E9502` SSM eval degradation — eval SSM MSE much higher than training SSM MSE. Severity Notice. Evidence: ratio. Suggests overfitting or distribution shift.

Each finding carries `cjc_locke::FingerprintId` content-addressed over `(detector config, ExperimentReport.replay_hash)` so the Locke composition layer can route them to belief axes deterministically.

**Tests (~10):**
- Detector + sink emit findings in the right code range
- Three threshold scenarios (above, at, below) for each finding
- Finding content-addressing is deterministic across runs
- Integration test: full `validate(df) → run_experiment → CronosRegimeShiftDetector` pipeline emits expected findings

### 3.2 Vault deep-docs (6 files)

The brief's original Phase 5 deep-doc list, deferred to Phase 6 because the architecture wasn't stable enough until 4c shipped. Each is ~150-400 lines.

| File | Content |
|---|---|
| `Cronos GAN Architecture.md` | Full Mermaid architecture, per-module responsibility table, the predictor/challenger framing, the determinism stack from Phase 1 to 4c |
| `State Space Model Primitive.md` | The structural stability argument (`‖A‖_F = α` exactly), the row-normalisation construction, the linearity+time-invariance choice, comparison to S4/Mamba (we explicitly don't claim better; we claim deterministic) |
| `Liquid Neural Network Primitive.md` | The sigmoid-scaled τ refactor from Phase 2, the gates-are-inspectable design choice, the overflow-safe sigmoid implementation, comparison to Hasani et al. 2020/2021 |
| `Adversarial Temporal Training.md` | The full NCL derivation showing our challenger loss = Negative Correlation Learning, the predictor-first ordering rationale, the alternating-update determinism contract, and the Phase 3b + Phase 4c structural-test analysis |
| `Cronos GAN Experiment Results.md` | The Phase 4b + Phase 4c sweep tables, the empirical flip, the per-mode recommendations, the multi-seed statistics from Phase 4d |
| `Cronos GAN Verification Report.md` | The full 131+ test summary, the cross-platform CI matrix results, the determinism contract checklist, what each test catches |

### 3.3 Python bridge via maturin

Mirror the `cjc-locke-py` pattern. `python/cjc_cronos_gan/` exposing:
- `CronosSeed`, `TemporalGanConfig.{symmetric, ssm_as_generator, liquid_as_generator}`
- `run_experiment`, `run_experiment_sweep`
- `ExperimentReport`, `ExperimentSweepReport`, `EvalReport`, `TemporalDisagreement` as Python dataclasses
- `format_table()` as a `__repr__`
- Bytes-identical replay across Rust and Python (the `replay_hash` should match a Python re-construction from the same `(config, seed)`)

The Python bridge does NOT expose the autodiff layer or the `Trainable` trait — those are Rust-only. Python users get the high-level `run_experiment_sweep` API and the structured reports.

**Tests (~10 pytest):**
- Each public function callable from Python
- Replay_hash matches between Rust and Python re-runs
- Format_table output identical
- Errors translate cleanly (no Rust panics → Python exceptions)

### 3.4 Risks / mitigations

- **Locke detector requires `cjc-cronos-gan` as a runtime dep of `cjc-locke`** — would create a circular reference. Mitigation: put the detector in `cjc-cronos-gan/src/locke_detector.rs` and have Locke load it dynamically OR document that the detector is owned by Cronos GAN, not Locke.
- **Vault docs are 6 files and ~1500 total lines of prose** — significant writing effort. Mitigation: parallel agents, with the main thread reviewing for consistency.
- **Python bridge needs PyO3 setup and maturin in CI** — there's precedent in `cjc-locke-py`. Mitigation: copy the pattern; the binding code itself is mechanical.

## 4. Open empirical questions Phase 4d → 6 should answer

| Question | Phase that resolves it |
|---|---|
| Is the Phase 4c flip stable across seeds? | 4d (n_seeds=10 statistics) |
| Does a decaying λ in `liquid_as_generator` give the best of both worlds? | 4d (LambdaSchedule::Linear sweep) |
| Does per-mode `lr` change the empirical hierarchy? | 4d |
| Do Bolero-fuzzed configs ever surface previously-unseen panics? | 5b |
| Does the cross-platform CI matrix actually find platform-specific divergence? | 5b |
| Is the regime-shift score correlated with downstream decisions when consumed by Locke? | 6 (custom detector integration test) |
| Does the Python API match Rust byte-for-byte? | 6 (replay_hash equivalence test) |

## 5. Recommended execution order

```
Phase 4d → Phase 5b → Phase 6
```

Rationale:

- **4d before 5b**: the multi-seed sweep results inform what to fuzz. If 4d shows the Phase 4c flip is unstable across seeds, the fuzz targets should explicitly cover the cells where the flip flipped back. If 4d shows the flip is robust, fuzz can focus on edge-case configs.
- **5b before 6**: the cross-platform CI matrix activation in 5b validates the byte-identity claim Locke will rely on. Shipping the Locke detector before the cross-platform tests run means publishing a determinism contract that hasn't been audited on multiple OSes.
- **6 last**: Phase 6 makes Cronos GAN *consumable* (Locke composition + Python). Doing this before the empirical findings stabilise (4d) and the determinism is cross-platform-verified (5b) is putting the deliverable in users' hands before the foundation is hardened.

If the user wants to compress: 4d + 5b can run in parallel on different worktrees (they don't touch the same code), with 6 strictly after both.

## 6. Branch state for the next session driver

- **Branch**: `feat/cjc-cronos-gan` at `ce2f0d3`
- **Local-only**: never pushed; CI matrix YAML committed but inactive
- **Worktree**: `C:/Users/adame/CJC/.claude/worktrees/cronos-gan/`
- **131 tests passing release**
- **`cargo run --example sweep --release`** produces the Phase 4c 15-cell table in ~7 seconds, demonstrating the empirical flip in a single command

Each phase is independently shippable on top of `ce2f0d3` without cleanup.

## See also

- [[Cronos GAN Phase 1 Overview]] — full architecture + Phases 1-4c with the empirical flip
- [[Phase 2 Handoff]] — the workspace-level handoff template this document mirrors
- [[ADR-0041]] (cjc-locke custom detector layer) — Phase 6's E9500..=E9999 reservation
- ADR-0045 (cjc-tempest) — same workspace, same determinism pattern, different application
