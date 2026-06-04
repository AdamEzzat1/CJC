---
title: "Cronos GAN Phase 1 Overview"
tags: [showcase, experimental, temporal, gan, ssm, liquid-nn, determinism]
status: "🧪 Experimental — Phase 1 SCAFFOLDING shipped"
crate: cjc-cronos-gan
version: 0.1.11
date: 2026-06-04
---

# Cronos GAN Phase 1 Overview

> [!warning] Experimental crate
> Cronos-GAN is currently an **experimental deterministic temporal modeling crate**. The first goal is correctness, auditability, and reproducibility — **not** state-of-the-art forecasting accuracy. Do not benchmark this crate against statsmodels, Prophet, N-BEATS, or other production forecasting systems until at least Phase 4 (experiment harness + adversarial training loop) lands. The point of Phase 1 is that the **primitives are wired correctly** and the **determinism contract is structurally inviolable**.

## What is Cronos GAN?

Cronos GAN (`cjc-cronos-gan`) is an experimental temporal adversarial system where two structurally opposite temporal models compete on the same time series:

- **State Space Model (SSM)** — *stable long-range latent dynamics adversary*. Linear, time-invariant, with spectral norm of the transition matrix `||A||_2 ≤ α < 1` **by construction**. Forgets perturbations exponentially.
- **Liquid Neural Network** — *adaptive nonlinear local dynamics adversary*. Discrete liquid time-constant network where the per-dimension time constant `τ` is gated by the current input *and* current state via clipped softplus. Slow (memory-like) when input is stationary; fast (reactive) when input is volatile.

The **disagreement** between these two networks on the same input sequence is the signal Cronos GAN ships in later phases — a large gap means the data has just transitioned between a smooth regime and a locally volatile one. That's the regime-shift / anomaly score the brief asks for.

This is **not** a generic GAN. Cronos GAN never trains on synthetic-vs-real classification. It trains stable latent dynamics against adaptive local dynamics, and the disagreement is the inspectable artifact.

> [!info] Sibling crate distinction
> Do not confuse `cjc-cronos-gan` (this crate, experimental adversarial temporal modeling) with [`cjc-cronos`](../03_Compiler/cjc-cronos%20overview.md) (classical deterministic forecasting: ETS, ARIMA, Kalman, STL). The sibling crate is feature-complete v0.1 and ready for crates.io; this crate is Phase 1 scaffolding for a multi-phase research roadmap.

## Architecture

```mermaid
flowchart TD
    seed[CronosSeed] -->|substream\n'ssm.A','ssm.B','ssm.C'| ssm_params[StateSpaceParams<br/>A, B, C, D]
    seed -->|substream\n'liquid.W_h','liquid.W_x',...| liq_params[LiquidParams<br/>W_h, W_x, b, W_τ, b_τ, W_out, b_out]

    inputs[TimeSeries / TemporalBatch] --> ssm_step[StateSpaceModel::step<br/>x' = A·x + B·u<br/>y = C·x + D·u]
    inputs --> liq_step[LiquidNetwork::step<br/>act = tanh(W_h·h + W_x·u + b)<br/>τ = clip(softplus(W_τ·[u;h] + b_τ))<br/>h' = h + dt/τ · (-h + act)<br/>y = W_out·h + b_out]

    ssm_params --> ssm_step
    liq_params --> liq_step

    ssm_step --> ssm_out[StateSpaceRolloutResult]
    liq_step --> liq_out[LiquidRolloutResult<br/>+ time_constants + gates]

    ssm_out -.->|Phase 3| dis[TemporalDisagreement<br/>ssm_score, liquid_score,<br/>absolute_gap, regime_shift_score]
    liq_out -.->|Phase 3| dis

    dis -.->|Phase 3+| trainer[TemporalGanTrainer]
```

Dashed edges mark Phase 3+ work (not shipped). Solid edges are what Phase 1 actually delivers.

### Why these two specifically

The structural opposition is the point.

| Property | SSM | Liquid NN |
|---|---|---|
| Linearity | Linear in `x` and `u` | Nonlinear (`tanh`, `softplus`) |
| Time-invariance | Time-invariant `A, B, C, D` | Time-varying effective time constant `τ(x, h)` |
| Memory regime | Exponential decay at rate `α` | Variable; slow when input stationary, fast otherwise |
| Inductive bias | Smooth continuations | Reactive to local volatility |
| Stability | Structural (`||A||_2 ≤ α` by construction) | Bounded (`τ` clipped to `[τ_min, τ_max]`) |
| State inspectable? | Yes (`StateSpaceState.x`) | Yes + `LiquidTimeConstant.tau` + `LiquidGate.gate` |

When the Phase 3 GAN trains, the disagreement score will become meaningful *because* these two architectures encode opposite biases. The SSM will under-react to spikes; the Liquid net will react too much to smooth segments. Their gap *is* a regime-shift signature.

## Determinism contract

Every random draw routes through `CronosSeed::substream(salt)`. Two distinct salt strings (e.g. `"ssm.A"` and `"liquid.W_h"`) derive independent SplitMix64 streams from the same master seed, so the SSM's transition matrix and the Liquid's recurrent matrix are *guaranteed not to share state* even when constructed from the same `CronosSeed`. This is enforced by an end-to-end test (`ssm_and_liquid_use_independent_rng_substreams`).

Specifically:

1. **RNG** — every draw through `cjc_repro::Rng` (SplitMix64), seeded from `CronosSeed.substream(salt)`.
2. **Reductions** — every matrix-vector product, row L² norm, and loss sum uses `cjc_repro::KahanAccumulatorF64`. Replaces accumulation-order dependence with order-independent Kahan compensation.
3. **Gate bounds** — Liquid `τ` is clipped into `[τ_min, τ_max]` by construction; `softplus` is overflow-safe at every finite f64 input via branching at ±20.
4. **Spectral bound** — SSM `A = (α/√state_dim) · R` where each row of `R` has unit L² norm. Frobenius norm `||A||_F = α` *exactly*, which implies `||A||_2 ≤ α < 1`. Test `transition_matrix_is_structurally_stable` asserts `|‖A‖_F − α| < 1e-12`.
5. **Iteration order** — `BTreeMap`/`BTreeSet` only, no `HashMap`. (None used yet in Phase 1.)
6. **No FMA, no thread-parallel reductions** in this crate.

## Public API (Phase 1)

```rust
use cjc_cronos_gan::{
    CronosSeed, LiquidConfig, LiquidNetwork, LiquidState,
    StateSpaceConfig, StateSpaceModel, StateSpaceState,
};

let seed = CronosSeed(42);

// Same input/output shape so disagreement is meaningful in Phase 3.
let ssm = StateSpaceModel::from_seed(StateSpaceConfig::new(8, 4, 2), seed).unwrap();
let liq = LiquidNetwork::from_seed(LiquidConfig::new(8, 4, 2), seed).unwrap();

// Roll both forward across the same input sequence.
let n_steps = 10;
let inputs: Vec<f64> = (0..n_steps * 4).map(|i| (i as f64 * 0.1).sin()).collect();
let ssm_out = ssm.rollout(&StateSpaceState::zeros(8), &inputs).unwrap();
let liq_out = liq.rollout(&LiquidState::zeros(8), &inputs).unwrap();

// Phase 3 will read (ssm_out.outputs, liq_out.outputs, liq_out.time_constants)
// to compute the TemporalDisagreement score.
```

### Primitives shipped

| Group | Types |
|---|---|
| Temporal core | `TimeStep`, `TimeSeries`, `TemporalBatch`, `SequenceMask`, `ForecastWindow`, `TemporalLoss` |
| State trait | `TemporalState`, `TemporalTransition<S>`, `TemporalRollout<S>` |
| SSM | `StateSpaceConfig`, `StateSpaceParams`, `StateSpaceState`, `StateSpaceModel`, `StateSpaceStepResult`, `StateSpaceRolloutResult` |
| Liquid NN | `LiquidConfig`, `LiquidParams`, `LiquidState`, `LiquidNetwork`, `LiquidTimeConstant`, `LiquidGate`, `LiquidStepResult`, `LiquidRolloutResult` |
| Determinism | `CronosSeed`, `CronosRunId` |
| Errors | `CronosGanError` (5 variants) |

## Test summary (Phase 1)

| Module | Tests | Coverage |
|---|---|---|
| `seed.rs` | 6 | substream determinism, salt independence, seed divergence, run-ID stability under seed/config perturbation |
| `time_series.rs` | 9 | construction, dimension/NaN rejection, batch shape consistency, mask length validation, forecast window OOB, MSE/MAE/Huber known values + invalid delta |
| `temporal_state.rs` | 1 | trait + rollout invariants |
| `ssm.rs` | 9 | config validation, seed determinism, seed divergence, `‖A‖_F = α` exactness, step shape, rollout length, bounded inputs ⇒ finite states (200 steps), byte-identical rollout across runs |
| `liquid.rs` | 8 | config validation, seed determinism, seed divergence, step shape + `τ` bounds, `τ` stays bounded under `±1e6` inputs, rollout length, byte-identical rollout across runs |
| `lib.rs` | 2 | SSM + Liquid share I/O shape; SSM and Liquid use **independent RNG substreams** even with same seed |
| **Total** | **35** | — |

All 35 unit tests pass on Phase 1 commit `<TBD>` in release profile. The 10-second doc test is the `lib.rs` quick-start example actually running a 10-step rollout end-to-end.

## Known limitations

- **No training loop** — neither network learns anything yet. Forward steps + rollouts only. Adversarial training, autodiff integration via `cjc_ad::GradGraph`, and the GAN layer (`TemporalGan`, `TemporalGanTrainer`, `TemporalDisagreement`) all ship in Phase 3.
- **No experiment harness** — the brief's five synthetic datasets (smooth sine, noisy sine, regime shift, step-change anomaly, chaotic spike) are not yet wired. Phase 4.
- **No property tests, no Bolero fuzz targets** — Phase 5. Phase 1's invariants are well-covered by deterministic unit tests; structural fuzzing is more valuable after the GAN layer adds branching.
- **No cross-platform CI matrix** — should be wired before Phase 4 ships, matching the Phase 2 handoff §1.5 expectation for `cjc-tempest`.
- **No `tests/cronos/{unit,integration,prop,fuzz}/` workspace-level layout** — inline `#[cfg(test)]` blocks only. Workspace-level layout ships once integration tests have something cross-module to integrate (i.e. after Phase 3 GAN).
- **No autodiff** — Phase 1 forward steps don't touch `cjc_ad::GradGraph`. Phase 2 adds autodiff integration once the v0.1 API stabilises.
- **Liquid τ uses fixed clip bounds** — no adaptive `τ_min`/`τ_max` learning. Acceptable for Phase 1; revisited in Phase 3.
- **No `cjc-locke` composition yet** — Phase 6 adds a custom-detector emitting regime-shift anomaly findings in the `E9500..=E9999` custom range.
- **Liquid output `y = W_out · h_prev + b_out`** uses the *previous* hidden state — matches SSM convention (`y_t = C x_t`) so the two networks' outputs are step-aligned for disagreement scoring. This is a design choice, not a bug; revisit if Phase 3 disagreement scoring needs `h_t` instead.

## Future phases

| Phase | Adds |
|---|---|
| **2** | Training-loss helpers, `cjc_ad::GradGraph` integration, per-network supervised training loop |
| **3** | `TemporalGanConfig`, `TemporalGan`, `TemporalGanTrainer`, `TemporalDisagreement`, three GAN modes (SSM-as-generator / Liquid-as-generator / symmetric), adversarial + reconstruction + forecast + temporal-consistency losses |
| **4** | Synthetic experiment harness (5 datasets), regime-shift score validation, deterministic replay hash on every run |
| **5** | `tests/cronos/{unit,integration,prop,fuzz}/` workspace layout, proptest suite, Bolero fuzz targets, vault deep-docs (Architecture, SSM Primitive, Liquid Primitive, Adversarial Training, Experiment Results, Verification Report) |
| **6** | `cjc-locke` custom detector for regime-shift findings (`E9500+`), Python bridge, lift `publish = false` for v0.2+ |

## See also

- ADR for Cronos GAN — not yet written; queued for Phase 3 once the GAN layer's design is concrete enough to commit
- Sibling crate: [`cjc-cronos`](../03_Compiler/cjc-cronos%20overview.md) (classical forecasting; do not confuse)
- Composition target: [Phase 2 Handoff](../10_Roadmap_and_Open_Questions/Phase%202%20Handoff.md) §5 — Locke v0.9 custom detector layer
- Determinism reference: [Determinism](../05_Determinism_and_Numerics/Determinism.md), [SplitMix64](../05_Determinism_and_Numerics/SplitMix64.md), [Kahan Summation](../05_Determinism_and_Numerics/Kahan%20Summation.md)
