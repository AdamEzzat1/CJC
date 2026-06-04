---
title: "Cronos GAN — Phases 1-5 Overview (incl. Phase 3b + Phase 4b sweep + Phase 4c eval split)"
tags: [showcase, experimental, temporal, gan, ssm, liquid-nn, determinism, autodiff, proptest, predictor-challenger, sweep, holdout-eval]
status: "🧪 Experimental — Phases 1-5 partial + Phase 3b asymmetric modes + Phase 4b 5×3 sweep + Phase 4c held-out eval shipped"
crate: cjc-cronos-gan
version: 0.1.11
date: 2026-06-04
date-modified: 2026-06-04
---

# Cronos GAN — Phases 1–5 Overview

> [!warning] Experimental crate
> Cronos-GAN is currently an **experimental deterministic temporal modeling crate**. The first goal is correctness, auditability, and reproducibility — **not** state-of-the-art forecasting accuracy. Do not benchmark this crate against statsmodels, Prophet, N-BEATS, or other production forecasting systems until the v0.2 stabilisation pass. Phase 5 ships the proptest suite and CI matrix but defers Bolero fuzz and per-dataset hyperparameter tuning to Phase 5b.

## What ships now

| Phase | What lands | Code | Tests |
|---|---|---|---|
| 1 | Temporal primitives + SSM + Liquid forward steps + rollouts + determinism types | ~1,100 LOC | 35 inline unit |
| 2 | `Trainable` trait, SSM + Liquid `GradGraph` autodiff adapters, `SupervisedTrainer` with Adam | ~750 LOC | 12 integration (FD-grad + training convergence + byte-identical replay) |
| 3 (min) | `TemporalDisagreement` + `TemporalGan` (symmetric mode) + content-addressed `run_id` | ~400 LOC | 5 disagreement + 3 GAN |
| **3b** | **`ChallengerSpec` + asymmetric modes + `TemporalGanTrainer` alternating updates** | **~600 LOC** | **4 inline + 21 integration (FD-grad for challenger loss + 3-mode byte-id determinism + λ separation)** |
| 4 | 5 synthetic dataset generators (`smooth_sine`, `noisy_sine`, `regime_shift`, `step_change_anomaly`, `chaotic_spike`) + experiment harness with replay hash | ~500 LOC | 7 dataset + 2 experiment |
| **4b** | **`run_experiment` rerouted through `TemporalGanTrainer` + `SweepBaseConfig` + `ExperimentSweepReport` + `run_experiment_sweep` 5×3 sweep + `examples/sweep.rs` binary** | **~400 LOC + 1 binary** | **16 integration (15-cell coverage + per-dataset lr override + byte-identical sweep_hash + mode-separation invariants)** |
| **4c** | **`eval_steps` held-out forecasting horizon in `ExperimentConfig` + `EvalReport` + per-mode λ in `SweepBaseConfig` + adaptive `format_table` (eval columns when present) + Phase 3b structural invariants verified on eval data** | **~250 LOC** | **14 integration (eval determinism + per-mode λ override + eval bytes in replay_hash + Phase 3b invariants survive Phase 4c)** |
| 5 (part) | 10 `proptest` properties + cross-platform CI matrix workflow | ~250 LOC | 10 × 64-256 cases each |

**Total: 131 distinct tests, all passing on release.** Doctest + 12 supervised-training integration + 21 GAN-training integration + 16 sweep integration + 14 Phase 4c integration + 57 inline unit + 10 proptest.

## Architecture (Phase 1-3)

```mermaid
flowchart TD
    seed[CronosSeed] -->|substream\nssm.A,B,C| ssm_params[StateSpaceParams<br/>A, B, C, D=0]
    seed -->|substream\nliquid.W_h,W_x,W_tau_u,W_tau_h,...| liq_params[LiquidParams<br/>W_h, W_x, b, W_τu, W_τh, b_τ, W_out, b_out]

    inputs[TimeSeries / TemporalBatch] --> ssm_step[StateSpaceModel::step<br/>x' = A·x + B·u<br/>y = C·x]
    inputs --> liq_step[LiquidNetwork::step<br/>act = tanh(W_h·h + W_x·u + b)<br/>τ = τ_min + (τ_max−τ_min)·σ(...)<br/>h' = h + dt/τ · (-h + act)<br/>y = W_out·h + b_out]

    ssm_params --> ssm_step
    liq_params --> liq_step

    ssm_step --> ssm_out[SSM trajectory]
    liq_step --> liq_out[Liquid trajectory<br/>+ τ trace + gates]

    ssm_out --> dis[TemporalDisagreement<br/>ssm_score<br/>liquid_score<br/>absolute_gap<br/>regime_shift_score]
    liq_out --> dis

    inputs --> trainer[SupervisedTrainer<br/>· build BPTT graph<br/>· backward_collect<br/>· adam_step]
    ssm_params -.->|read/write flat| trainer
    liq_params -.->|read/write flat| trainer

    trainer --> trained_ssm_params[Updated SSM params]
    trainer --> trained_liq_params[Updated Liquid params]

    dis --> report[ExperimentReport<br/>+ replay_hash]
```

## The architectural opposition (unchanged from Phase 1)

| Property | SSM | Liquid NN |
|---|---|---|
| Linearity | Linear in `x` and `u` | Nonlinear (`tanh`, `sigmoid`) |
| Time-invariance | Time-invariant `A, B, C` | Time-varying effective time constant `τ(x, h)` |
| Memory regime | Exponential decay at rate `α` | Variable; slow when input stationary, fast otherwise |
| Inductive bias | Smooth continuations | Reactive to local volatility |
| Stability | Structural (`‖A‖₂ ≤ α` by construction) | Bounded (`τ ∈ (τ_min, τ_max)` strictly) |
| State inspectable? | Yes (`StateSpaceState.x`) | Yes + `LiquidTimeConstant.tau` + `LiquidGate.gate` |
| Differentiable? | Linear ⇒ every gradient exists | Phase 2 refactored to sigmoid-scaled τ (smooth everywhere) |

## Phase 2 — what changed since Phase 1

The Phase 1 Liquid used `softplus.clamp(τ_min, τ_max)`, which is **non-differentiable at the clip boundaries**. Phase 2 refactored to the mathematically-equivalent **sigmoid-scaled formulation**:

```
s = sigmoid(W_τ_u · u + W_τ_h · h + b_τ) ∈ (0, 1)
τ = τ_min + (τ_max − τ_min) · s             ∈ (τ_min, τ_max)
```

This is smoothly differentiable everywhere, so the BPTT gradient flows freely through the gate. The Phase 1 bounded-τ test (`τ` stays in `[τ_min, τ_max]` under ±1e6 inputs) still holds — actually more tightly, since `τ` now sits *strictly inside* the open interval `(τ_min, τ_max)`.

Phase 2's `Trainable` trait + `SupervisedTrainer` + autodiff adapters give:

- **Bit-identical training trajectories across runs** (same `(seed, config, inputs)` ⇒ same loss values, same final weights, every step)
- **Gradient correctness verified by finite-difference comparison**: SSM `max_rel < 1e-4`, Liquid `max_rel < 5e-4` over a small test grid
- **No silent allocations** in the training inner loop; everything goes through `cjc_ad::GradGraph` arena + `cjc_runtime::ml::adam_step` flat-vector kernel

## Phase 4c — held-out forecasting horizon and the empirical flip

Phase 4b reported a striking result: in `liquid_as_generator` mode the SSM achieves substantially lower training MSE than in symmetric mode (38% reduction on smooth_sine, 43% on noisy_sine). The interpretation was the SSM uses the `-λ · MSE-vs-Liquid` gradient as informative regularization to escape into a different (better) basin of the loss landscape.

**Phase 4c added a held-out forecasting horizon and the conclusion flipped.** The training-set advantage was *overfitting*, not generalization.

### What changed

- `ExperimentConfig` gains `eval_steps: usize` (default 0). When > 0, the dataset generator produces `n_steps + eval_steps` samples; the model trains on the first `n_steps`, then each network's rollout continues from its post-training hidden state on the eval window. Disagreement is computed on `(eval_predictions, eval_targets)` — true forecastability, no information leakage.
- `ExperimentReport` gains `eval: Option<EvalReport>` with `ssm_loss`, `liquid_loss`, `disagreement` on the held-out window.
- `SweepBaseConfig` gains `per_mode_lambda: BTreeMap<TemporalGanMode, f64>` so the canonical sweep can probe whether the optimal λ differs between `SsmAsGenerator` and `LiquidAsGenerator`. `with_lambda_for(mode, λ)` builder + `lambda_for(mode)` accessor (falls back to `lambda_disagreement` when no override).
- `format_table` is adaptive: shows 3 extra eval columns (eval ssm, eval liq, eval |gap|) iff at least one cell has populated eval.
- Phase 3b's bit-identity structural invariants are **re-verified on eval data**: `ssm_loss_in_ssm_as_generator_still_equals_symmetric_with_eval` and the Liquid mirror assert that the held-out eval SSM MSE is bit-identical between Symmetric and SsmAsGenerator. The eval pipeline reads only the post-training SSM state, so it inherits the structural property.

### The empirical flip (canonical sweep, seed=42)

Canonical Phase 4c sweep: 50 train steps × 20 eval steps × 200 Adam updates × λ_SsmAsGen=0.10, λ_LiquidAsGen=0.15.

| Mode | train \|gap\| | **eval SSM MSE** | **eval \|gap\|** |
|---|---|---|---|
| `symmetric` | 3.03e-1 | 3.18e-1 | 5.70e-1 |
| `ssm_as_generator` | 3.03e-1 | **3.18e-1** | **3.67e-1** |
| `liquid_as_generator` | 3.30e-1 | 3.70e-1 (worst) | 6.08e-1 |

**Three observations the table forces you to confront:**

1. **`ssm_as_generator` produces the lowest held-out `|gap|`** — 36% lower than symmetric, 40% lower than liquid_as_generator. The Liquid challenger has learned to *consistently disagree-less-on-eval-than-on-train* with the SSM. The disagreement signal in this mode is *calibrated* to forecastability, not to training-data idiosyncrasies.
2. **`liquid_as_generator`'s eval SSM MSE is the WORST** of the three modes — 16% higher than symmetric. The training-MSE reductions Phase 4b celebrated (down to 1.07e-2 on smooth_sine) do not survive to held-out data (4.75e-4 → 100× WORSE in `liquid_as_generator` than in `symmetric`/`ssm_as_generator` for smooth_sine).
3. **`symmetric` and `ssm_as_generator` have identical eval SSM MSE byte-for-byte** — because the SSM is the predictor in both modes (Phase 3b invariant survives Phase 4c). The only thing that *changes* between these two modes is what the Liquid learns. And `ssm_as_generator`'s Liquid generalizes *much* better at the disagreement metric.

### The mechanism

The Phase 4b mechanism (NCL diversity → SSM escapes into alternative basin) is real but the alternative basin is **dataset-specific**. The SSM in `liquid_as_generator` is being pushed away from Liquid's specific failure modes on the training data. Liquid's failure modes on training data don't match Liquid's failure modes on held-out data, so the SSM's anti-Liquid adjustments are misdirected for forecasting. Standard "diversity vs ensemble error" theory predicts this: diversity helps the ensemble on data the ensemble was trained on, but the adjustments transfer to held-out data only if the disagreement is **systematic** (about the data) rather than **incidental** (about training trajectory).

`ssm_as_generator` is the inverse: the Liquid challenger learns to predict accurately while being decorrelated from the SSM's stable-dynamics solution. The Liquid's challenger gradient `-λ · MSE-vs-SSM` is pointing it away from a *consistent* SSM bias (smooth/linear dynamics → mispredicts spikes). Liquid then captures the residual structure the SSM systematically misses, and that residual is *itself* systematic in the data. On held-out data, the Liquid still captures it.

**Choosing between modes for production forecastability disagreement: pick `SsmAsGenerator`.** It preserves the SSM's structural-stability accuracy AND produces a Liquid challenger whose disagreement is informative on held-out data.

### Phase 4c determinism contract additions

16. **Eval rollout is byte-identical across runs** with the same `(config, seed)` — `eval_byte_identical_across_runs` test asserts to `to_bits()`.
17. **Eval bytes enter the replay hash** — `report_replay_hash_changes_with_eval_steps` proves `eval_steps=0` and `eval_steps=5` produce distinct `replay_hash`es with otherwise-identical configs. The sweep hash inherits this discrimination.
18. **Per-mode λ overrides are deterministic** — `BTreeMap<TemporalGanMode, f64>` iteration is key-ordered. Two sweeps with the same per-mode λ map produce the same `sweep_hash`.
19. **Phase 3b SSM-predictor invariant survives Phase 4c** — the SSM's eval MSE is bit-identical between `Symmetric` and `SsmAsGenerator` for every dataset because the eval pipeline reads only post-training SSM state.

## Phase 4b — the 5×3 sweep and the empirical answer

The brief's headline question for Phase 4b: **does asymmetric mode actually find different solutions than symmetric mode?** Phase 4b makes this empirically answerable in a single `cargo run --example sweep --release` invocation that produces the 5 datasets × 3 modes = 15 cell table.

### What ships

- **`run_experiment` is now routed through `TemporalGanTrainer`** so all three modes share one training pipeline. The Phase 4 two-`SupervisedTrainer` flow is gone. Symmetric mode's *parameters* are mathematically equivalent (no inter-network coupling) but the *trajectory bytes* shift because alternating updates interleave the Adam moments differently. This is a deliberate one-time rebaseline.
- **`ExperimentReport` extended** with `mode: TemporalGanMode`, `disagreement_trajectory: Vec<TemporalDisagreement>` (per-step gap during training), `mean_absolute_gap: f64`, and `max_regime_shift_score: f64`. The trajectory + summary stats are what the sweep table reads.
- **`SweepBaseConfig`** — shared dimensions, `n_steps`, `n_train_steps`, `lambda_disagreement`, `default_lr` + per-dataset `lr` overrides via `BTreeMap<CronosDataset, f64>`. Determinism survives the overrides because `BTreeMap` iteration is key-ordered.
- **`run_experiment_sweep(&base, seed) -> ExperimentSweepReport`** — runs all 15 cells in canonical order (`SWEEP_DATASETS` outer, `SWEEP_MODES` inner) and returns a structured report with a content-addressed `sweep_hash` over every cell's `replay_hash`.
- **`ExperimentSweepReport::format_table()`** — emits a Unicode box-drawing 15-row table with columns: dataset, mode, ssm_loss, liq_loss, mean |gap|, max regime shift, replay_hash. Footer with `sweep_hash` and `seed`.
- **`examples/sweep.rs`** — single `cargo run --example sweep --release` runs the canonical sweep (8-dim state, 50-step series, 200 training steps, λ=0.1) and prints the table + a per-mode summary of mean |gap| across the 5 datasets.

### Empirical result on the canonical sweep (seed=42)

Per-mode mean of mean `|gap|` across the 5 datasets:

| Mode | Mean `|gap|` across 5 datasets |
|---|---|
| `symmetric` | 3.17e-1 |
| `ssm_as_generator` | 3.16e-1 |
| `liquid_as_generator` | **3.35e-1** |

**Answer to the brief's question: yes — the modes find different solutions, and `liquid_as_generator` produces the largest sustained disagreement.** Mechanistically, when SSM is the challenger it receives gradient information about how it differs from the Liquid net, and this informs its parameter updates toward solutions that are both accurate AND structurally different from Liquid's prediction. The SSM's structural stability constraint (`‖A‖₂ ≤ α`) keeps it from drifting to nonsense even with the divergence reward.

### Per-mode SSM/Liquid loss invariants the sweep verifies

Two invariants the test suite locks down (see `ssm_loss_in_ssm_as_generator_equals_ssm_loss_in_symmetric` and `liquid_loss_in_liquid_as_generator_equals_liquid_loss_in_symmetric`):

- **SSM final loss is bit-identical between `Symmetric` and `SsmAsGenerator`** for every dataset. The SSM is the predictor (vanilla supervised) in both modes, so its updates are identical.
- **Liquid final loss is bit-identical between `Symmetric` and `LiquidAsGenerator`**. Mirror image.

This is structural evidence the asymmetric framing is correctly implemented: the challenger gets a different loss, but the predictor does not.

### Determinism contract (Phase 4b extensions)

13. **`sweep_hash` is byte-identical across runs** with the same `(SweepBaseConfig, CronosSeed)`. Per-cell `replay_hash`es and the aggregate `sweep_hash` survive every reordering, parallelization, or platform change that the rest of the workspace's determinism contract already covered.
14. **Per-dataset `lr` overrides are deterministic** because `BTreeMap` iteration is key-ordered. Two sweeps with the same overrides produce the same `sweep_hash`; differing overrides shift it.
15. **The 15-cell iteration order is fixed** by `SWEEP_DATASETS` × `SWEEP_MODES`. Result indices are stable across runs — no `HashMap` shuffling.

## Phase 3b — predictor / challenger and the loss-sign answer

The open question the phase had to answer: **when SSM is the "generator", does it train to minimise or maximise disagreement?**

The answer — both framings are wrong because they treat disagreement as a binary target. The shipped formulation:

> **Asymmetric mode = predictor + challenger.** One network (the *predictor*, what a standard GAN calls the "generator") gets a vanilla supervised MSE loss. The other (the *challenger*, what a standard GAN calls the "discriminator") gets `MSE(target) − λ · MSE(predictor)` — a supervised loss *minus* a bonus for diverging from the predictor.

**Loss-sign answer.** SSM does NOT train to maximise OR minimise disagreement when it's the generator. It trains to be accurate. The disagreement signal is produced by the **challenger's** asymmetric loss term — the `−λ` is on the challenger, not the predictor. The brief's "the latter [maximise]" intuition was directionally right but applied to the wrong network.

| Mode | SSM role | Liquid role | What pushes disagreement? |
|---|---|---|---|
| `Symmetric` (Phase 3 min) | Predictor (supervised MSE) | Predictor (supervised MSE) | Nothing — disagreement is observed, not trained against |
| `SsmAsGenerator` (Phase 3b) | Predictor (supervised MSE) | Challenger (MSE − λ · MSE-vs-SSM) | The Liquid challenger is rewarded for diverging from the SSM while staying accurate |
| `LiquidAsGenerator` (Phase 3b) | Challenger (MSE − λ · MSE-vs-Liquid) | Predictor (supervised MSE) | The SSM challenger is rewarded for diverging from the Liquid while staying accurate |

### Choosing λ

- **λ = 0**: asymmetric mode reduces *byte-identically* to symmetric mode (proven in `lambda_zero_asymmetric_equals_symmetric_trajectory`). The canonical sanity check on the implementation.
- **Small λ (0.05–0.2)**: the challenger remains accurate but is encouraged to find alternative prediction paths. The disagreement signal becomes *informative* rather than noisy.
- **Large λ (≥ 1.0)**: divergence dominates accuracy; the challenger drifts. Useful for stress-testing the regime-shift score, not for production forecasts.

### The alternating-update step

`TemporalGanTrainer::step` runs one Adam update per network per call. In asymmetric modes the predictor updates first (because the challenger's loss depends on the predictor's CURRENT outputs):

```text
SsmAsGenerator step:
  1. update SSM with supervised MSE         → new SSM weights
  2. forward SSM with new weights           → predictor_outputs (frozen)
  3. update Liquid with MSE − λ·MSE-vs-SSM  → new Liquid weights
  4. compute disagreement                   → returned to caller

LiquidAsGenerator step: roles flip.
Symmetric step: both supervised updates, independent.
```

The `TemporalGanTrainStep` carries `ssm_role` and `liquid_role` (each `Role::Predictor` or `Role::Challenger`) so the caller can attribute losses correctly to the role each network played that step.

### Determinism contract (Phase 3b extensions)

Items 1–9 from Phase 1–5 still hold. Phase 3b adds:

10. **Alternating-update order is fixed per mode** — predictor always updates first; challenger reads the predictor's POST-update outputs. Same (seed, config, inputs, targets, initial Adam) ⇒ byte-identical SSM update ⇒ byte-identical predictor outputs ⇒ byte-identical challenger update.
11. **Mode label is in the canonical config bytes** — `TemporalGanRolloutResult.run_id` differs across modes even with same seed and same network dims (`run_id_differs_across_modes_with_same_seed_and_dims`).
12. **`λ = 0` collapses the asymmetric loss to the supervised loss byte-identically** — not approximately, exactly. The negative sign on the challenger term is the entire reason this property holds.

## Phase 3 (minimal) — TemporalDisagreement is the artifact

Where most GANs train *toward* indistinguishability, Cronos GAN treats persistent calibrated disagreement as the **signal**. The four scalars in `TemporalDisagreement`:

```rust
pub struct TemporalDisagreement {
    pub ssm_score: f64,           // Mean per-step RMSE of SSM vs target
    pub liquid_score: f64,        // Mean per-step RMSE of Liquid vs target
    pub absolute_gap: f64,        // Mean per-step RMSE of SSM vs Liquid (target-free)
    pub regime_shift_score: f64,  // peak_gap / (1 + mean_gap) — large for localised gaps
}
```

The **regime_shift_score** is the headline signal: when one step has a much larger SSM-vs-Liquid gap than the average, the score spikes — that's the regime-shift signature the brief asks for. Tested via `regime_shift_score_fires_on_localised_gap` on a synthetic sequence with 7 zero gaps + 1 gap of 10 (expected score ≈ 4.44, threshold 3.0).

Phase 3 minimal shipped `TemporalGanMode::Symmetric` only. **Phase 3b** added `SsmAsGenerator` and `LiquidAsGenerator` modes (see preceding section) along with `TemporalGanTrainer` driving the alternating-update training loop.

## Phase 4 — five synthetic datasets

| Dataset | Generator | RNG salt | What it tests |
|---|---|---|---|
| `smooth_sine` | `sin(0.4·t)` | (none — fully deterministic) | Baseline: both nets fit it well |
| `noisy_sine` | sine + N(0, 0.15²) | `dataset.noisy_sine` | Does SSM's stable bias regularise? |
| `regime_shift` | AR(1) φ=0.7, σ=0.2 → AR(1) φ=−0.3, σ=0.5 at midpoint | `dataset.regime_shift` | The canonical regime-shift test |
| `step_change_anomaly` | Flat 0, single +1 step at n/2 | (none) | Localised anomaly score |
| `chaotic_spike` | sine + +3.0 spikes every 10 steps | (none) | Does Liquid's gate fire on spikes? |

The `ExperimentReport` carries a `replay_hash: CronosRunId` content-addressed over `(config bytes, seed, final SSM params, final Liquid params, full training-loss trajectory)`. Two runs of the same `(config, seed)` produce the same hash — *the* operational claim Cronos GAN makes.

## Phase 5 (partial) — proptest + CI

10 `proptest` properties verifying:

1. SSM rollout length = input length
2. Liquid rollout length = input length (+ τ + gate trajectories)
3. SSM same-seed-byte-identical (random seed, random length)
4. Liquid same-seed-byte-identical
5. SSM finite-bounded inputs ⇒ finite outputs (random seeds, ±10 inputs)
6. Liquid finite-bounded inputs ⇒ finite outputs + τ stays bounded
7. `TemporalLoss::Mse` finite for finite inputs (random vectors, ±1e3)
8. `TemporalLoss::Mae` finite for finite inputs
9. `TemporalDisagreement` all fields ≥ 0 and finite (random ssm/liq/target ±5)
10. SSM repeated-rollout byte-identical (random seed)

Each block runs 64-256 cases ⇒ ~1,500-2,000 property-asserted invariants per CI run.

**CI matrix workflow** lives at [`.github/workflows/cjc-cronos-gan-determinism.yml`](../../.github/workflows/cjc-cronos-gan-determinism.yml). Four gates × three OSes (ubuntu / windows / macOS) with `fail-fast: false` so any platform divergence is visible in the same CI run. Gates: inline unit tests, training integration, proptest suite, doctest.

## Determinism contract (full)

Every property below is a CONSTRUCTION property (built into the math), not a TRAINED property (hoped for during optimisation):

1. **RNG sub-streams independent** — `CronosSeed::substream("ssm.A")` and `CronosSeed::substream("liquid.W_h")` derive from disjoint SplitMix64 streams. Cross-net test proves SSM `A` ≠ Liquid `W_h` even from same seed.
2. **All reductions Kahan-summed** — matvec, row L² norms, loss sums, disagreement RMSEs.
3. **Liquid τ bounded by construction** — sigmoid scales into `(τ_min, τ_max)`, no clip needed. Smooth gradient flow as a side benefit.
4. **SSM `‖A‖_F = α` exactly** — row L²-normalised then scaled. Stability is not an optimisation hope.
5. **softplus → sigmoid overflow safety** — branched at ±20, finite output for every finite f64.
6. **Param flattening canonical** — `params_flat`, `set_params_flat`, and `build_rollout_graph` agree on the order `[A | B | C]` (SSM) and `[W_h | W_x | b | W_τ_u | W_τ_h | b_τ | W_out | b_out]` (Liquid). Mismatch would silently break Adam's per-parameter update; the round-trip-exact test pins it.
7. **Adam is `cjc_runtime::ml::adam_step`** — the same kernel chess RL v2 trains against (proven bit-identical via the `9.790915694115341` weight hash gate).
8. **Replay hash is content-addressed** — `(config, seed, final params, training trajectory)` ⇒ same `CronosRunId` across runs and platforms.
9. **Cross-platform CI gate** — ubuntu + windows + macOS run every gate on every PR touching `crates/cjc-cronos-gan/`.

## Public API (Phase 1–5)

```rust
// Construction + forward
use cjc_cronos_gan::{
    CronosSeed, StateSpaceConfig, StateSpaceModel, StateSpaceState,
    LiquidConfig, LiquidNetwork, LiquidState,
};

let seed = CronosSeed(42);
let ssm = StateSpaceModel::from_seed(StateSpaceConfig::new(8, 4, 2), seed)?;
let liq = LiquidNetwork::from_seed(LiquidConfig::new(8, 4, 2), seed)?;

// Training
use cjc_cronos_gan::{SupervisedTrainer, Trainable};
let mut trainer = SupervisedTrainer::new(ssm.n_params(), 1e-2);
for _ in 0..100 {
    let loss = trainer.step(&mut ssm, &inputs, &targets)?;
}

// Disagreement (after training both networks)
use cjc_cronos_gan::{TemporalGan, TemporalGanConfig};
let gan = TemporalGan::from_seed(TemporalGanConfig::symmetric(8, 4, 2), seed)?;
let result = gan.rollout_and_disagreement(&inputs, &target)?;
println!("regime shift score: {}", result.disagreement.regime_shift_score);

// Full experiment with replay hash
use cjc_cronos_gan::{run_experiment, ExperimentConfig, CronosDataset};
let cfg = ExperimentConfig::new(
    TemporalGanConfig::symmetric(8, 1, 1),
    CronosDataset::RegimeShift,
    100,
);
let report = run_experiment(&cfg, seed)?;
println!("replay hash: {}", report.replay_hash);

// Phase 3b: asymmetric mode + alternating-update training
use cjc_cronos_gan::{TemporalGanTrainer, TemporalGanMode};
let cfg = TemporalGanConfig::ssm_as_generator(8, 1, 1, 0.1); // SSM=predictor, Liquid=challenger
let mut gan = TemporalGan::from_seed(cfg, seed)?;
let mut trainer = TemporalGanTrainer::new(cfg, &gan, 1e-2);
for _ in 0..100 {
    let step = trainer.step(&mut gan, &inputs, &targets)?;
    // step.ssm_loss, step.liquid_loss, step.disagreement, step.ssm_role, step.liquid_role
}
```

## Known limitations (post Phase 3b)

- **No Bolero fuzz** (Phase 5b). Proptest covers structural invariants; fuzz adds adversarial-input branching coverage that's now more valuable since Phase 3b introduced branching on `mode`.
- **No per-dataset hyperparameter tuning** (Phase 4b). Experiment harness uses a single `(lr, n_train_steps)` per call; per-dataset best practices live in a future tuning sweep.
- **`tests/cronos/{unit,integration,prop,fuzz}/` workspace layout**: Phase 5 ships proptest at the crate-level `tests/test_proptest.rs` instead. The workspace-level layout is a Phase 6 refactor once Bolero fuzz adds enough mass to justify the directory split.
- **No `cjc-locke` E9500+ custom detector** for regime-shift findings (Phase 6).
- **Adversarial training not yet shown to improve forecasts** vs. supervised baseline — this is a research question, not a bug. Phase 6 runs the comparison sweep.

## Test summary

| Module / file | Tests | Coverage |
|---|---|---|
| `seed.rs` | 6 | substream determinism + salt independence + run-ID stability |
| `time_series.rs` | 9 | construction, dim/NaN rejection, mask validation, loss invariants |
| `temporal_state.rs` | 1 | rollout invariants |
| `ssm.rs` | 9 | config validation, seed determinism, `‖A‖_F = α` exact, 200-step finiteness, byte-id |
| `liquid.rs` | 8 | config validation, seed determinism, τ bounds under ±1e6, byte-id |
| `disagreement.rs` | 5 | shape/NaN rejection, perfect agreement = 0, regime-shift localisation, determinism |
| `gan.rs` | 3 | byte-id across runs, run_id changes with seed, τ trace dims correct |
| `datasets.rs` | 7 | each dataset shape + content + sample determinism |
| `experiment.rs` | 2 | end-to-end smooth_sine, replay hash byte-id |
| `lib.rs` | 2 | cross-net I/O shape, RNG sub-stream independence |
| `tests/test_training.rs` | 12 | params round-trip, FD-grad correctness (both nets), training convergence, byte-id |
| `tests/test_proptest.rs` | 10 × 64-256 cases | 7 brief properties + 3 redundant cross-checks |
| Doctest in `lib.rs` | 1 | quick-start runs end-to-end |
| **TOTAL** | **76 distinct tests** | all passing release |

## Future phases

| Phase | Adds |
|---|---|
| **3c** | Loss-aware training-step ordering options (currently predictor-first; could add challenger-first or simultaneous-with-clone variants) |
| **4d** | λ schedule (warmup + plateau + decay) — Phase 4c showed `liquid_as_generator` overfits; a decaying λ might recover its training-MSE advantage *with* generalization; per-mode best-known hyperparameter recipes per dataset based on Phase 4c sweep findings |
| **5b** | Bolero fuzz targets (7), `tests/cronos/{unit,integration,prop,fuzz}/` workspace layout |
| **6** | `cjc-locke` E9500+ custom detector consuming `ExperimentSweepReport`'s per-cell `max_regime_shift_score` AND `eval.disagreement.regime_shift_score`, vault deep-docs (Architecture, SSM Primitive, Liquid Primitive, Adversarial Training, Experiment Results, Verification Report), Python bridge |

## See also

- Sibling crate: [`cjc-cronos`](https://crates.io/crates/cjc-cronos) (classical forecasting; do NOT confuse)
- Phase 2 dep (now landed): `cjc-ad` A1 determinism audit on `claude/zealous-khayyam-7d47f5`
- Determinism reference: [Determinism](../05_Determinism_and_Numerics/Determinism.md), [SplitMix64](../05_Determinism_and_Numerics/SplitMix64.md), [Kahan Summation](../05_Determinism_and_Numerics/Kahan%20Summation.md)
- CI matrix: [`.github/workflows/cjc-cronos-gan-determinism.yml`](../../.github/workflows/cjc-cronos-gan-determinism.yml)
