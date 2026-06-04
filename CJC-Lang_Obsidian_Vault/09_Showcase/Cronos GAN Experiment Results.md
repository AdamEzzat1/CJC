---
title: "Cronos GAN — Experiment Results"
tags: [showcase, experimental, results, sweep, holdout-eval, empirical-flip, multi-seed, n10]
status: "🧪 Phase 4d statistical validation complete — empirical flip is robust"
crate: cjc-cronos-gan
version: 0.1.11
date: 2026-06-04
date-modified: 2026-06-04
---

# Cronos GAN — Experiment Results

The empirical findings from the Phase 4b 15-cell sweep, the Phase 4c held-out evaluation discovery, and the Phase 4d multi-seed statistical validation. This is the doc where "what the model actually does" lives.

> [!info] Companion notes
> - [[Cronos GAN Architecture]] — the pipeline these experiments run on
> - [[Adversarial Temporal Training]] — the three modes and the loss
> - [[Cronos GAN Verification Report]] — what tests gate these findings

## The five synthetic datasets

All datasets are 1-D scalar time series with `input = output = 1`. They're deliberately simple — the contest is between inductive biases, not between architecture capacities.

| Dataset | Generator | Signal | Hardest for |
|---|---|---|---|
| `smooth_sine` | `sin(0.1 · t)` | Pure low-frequency oscillation | Neither — both win easily |
| `noisy_sine` | `sin(0.1 · t) + N(0, 0.1)` | Sine with white noise | Liquid (gates over-react to noise) |
| `regime_shift` | `sin(low_freq) → sin(high_freq)` at midpoint | One sharp frequency change | SSM (linear memory has wrong scale on both sides) |
| `step_change_anomaly` | Constant → constant + step at midpoint | One discrete level change | SSM (smooth-decay assumption violated) |
| `chaotic_spike` | Logistic-map seeded chaos | High-frequency unpredictable | Both — but Liquid handles the spike better |

## Phase 4b — the 15-cell sweep

Phase 4b shipped `run_experiment_sweep(base_config, seed)` running 5 datasets × 3 modes = 15 experiments, each at a single seed. The headline finding from the original sweep at `state_dim=8, n_steps=50, n_train_steps=200`:

| Dataset | Mode | Final SSM loss | Final Liquid loss | Mean &#124;gap&#124; |
|---|---|---|---|---|
| smooth_sine | symmetric | 0.0174 | 0.0345 | 0.38 |
| smooth_sine | ssm_as_gen | 0.0174 | 0.0369 | 0.36 |
| smooth_sine | liquid_as_gen | 0.0072 | 0.0345 | 0.43 |
| noisy_sine | symmetric | 0.0656 | 0.2712 | 0.41 |
| noisy_sine | ssm_as_gen | 0.0656 | 0.2357 | 0.40 |
| noisy_sine | liquid_as_gen | 0.0211 | 0.2712 | 0.46 |
| regime_shift | symmetric | 0.1042 | 0.0914 | 0.16 |
| regime_shift | ssm_as_gen | 0.1042 | 0.0840 | 0.17 |
| regime_shift | liquid_as_gen | 0.0929 | 0.0914 | 0.17 |
| step_change | symmetric | 0.0471 | 0.0252 | 0.18 |
| step_change | ssm_as_gen | 0.0471 | 0.0243 | 0.19 |
| step_change | liquid_as_gen | 0.0425 | 0.0252 | 0.19 |
| chaotic_spike | symmetric | 0.7463 | 0.7897 | 0.38 |
| chaotic_spike | ssm_as_gen | 0.7463 | 0.7575 | 0.39 |
| chaotic_spike | liquid_as_gen | 0.7045 | 0.7897 | 0.40 |

(Numbers from `cargo run --release --example sweep` at the time of Phase 4c.)

The structural invariants visibly hold (SSM losses match between Symmetric and SsmAsGenerator; Liquid losses match between Symmetric and LiquidAsGenerator — see [[Adversarial Temporal Training]] for why). The interesting per-mode question: which mode produces a *useful* disagreement signal?

Phase 4b's answer was inconclusive. The training-MSE was lower for `liquid_as_generator` on multiple datasets, suggesting it might be the winning mode — but that just measures fit to the *training* data.

## Phase 4c — the empirical flip

Phase 4c added held-out evaluation: train on `[0, n_steps)`, evaluate the trained networks on `[n_steps, n_steps + eval_steps)` of the same dataset's generated trajectory. Each network's state is rolled forward through the train portion, then continues from that state into the eval window — no information leakage, true forecastability.

The eval pass produces a separate `EvalReport` per cell:

```
eval.ssm_loss        — SSM MSE on the held-out window
eval.liquid_loss     — Liquid MSE on the held-out window
eval.disagreement    — TemporalDisagreement on the held-out window
```

And it flipped the Phase 4b conclusion. From the same `cargo run --example sweep` (now with `--with-eval-steps 20`):

**Per-mode means across all 5 datasets:**

| Mode | train &#124;gap&#124; | eval SSM MSE | eval &#124;gap&#124; |
|---|---|---|---|
| symmetric | 0.30 | 0.32 | 0.57 |
| **ssm_as_generator** | **0.30** | **0.32** | **0.37** |
| liquid_as_generator | 0.33 | 0.37 | 0.61 |

The winning mode on eval-disagreement is `ssm_as_generator` — by a significant margin (0.37 vs. 0.57 / 0.61). The mode that looked best on training MSE (`liquid_as_generator`) was overfitting: its lower training loss didn't transfer to the held-out window, and its eval disagreement is the WORST of the three.

The Phase 4c interpretation:
> **The asymmetric divergence pressure has to be applied to the network with the more flexible inductive bias.**
> SSM-as-predictor + Liquid-as-challenger works because the Liquid (nonlinear, time-varying) has the capacity to find divergent-but-accurate solutions; the SSM (linear, time-invariant) acts as a stable anchor.
> Liquid-as-predictor + SSM-as-challenger doesn't work because the SSM lacks the capacity to find a meaningfully-different-but-still-accurate alternative.

The "empirical flip" — what Phase 4b suggested vs. what Phase 4c discovered — is documented in the commit message of `ce2f0d3`.

## Phase 4d — the statistical validation

A single-seed result is a hypothesis, not a finding. Phase 4d added multi-seed sweeps via `SweepBaseConfig.n_seeds` + `seed_stride`, with `CellAggregate` carrying Kahan-summed mean + Welford sample variance across seeds.

The 4 `#[ignore]`'d empirical-flip-replication tests in `tests/test_phase_4d.rs` use `n_seeds = 10` and assert (with 10% slack):

| Claim | Result (n_seeds=10) | Ratio | Slack used |
|---|---|---|---|
| `eval_ssm_loss(SsmAsGen) ≤ 1.10 × eval_ssm_loss(LiqAsGen)` | SsmAsGen=0.710, LiqAsGen=1.825 | 0.389 | **none** — 61% margin |
| `eval_absolute_gap(SsmAsGen) ≤ 1.10 × eval_absolute_gap(Symmetric)` | SsmAsGen=0.929, Symmetric=0.958 | 0.969 | **none** — 3% margin |
| `eval_absolute_gap(SsmAsGen) ≤ 1.10 × eval_absolute_gap(LiqAsGen)` | SsmAsGen=0.929, LiqAsGen=1.149 | 0.808 | **none** — 19% margin |

All three inequalities held at strict `<`, not even needing the 10% slack. The Phase 4c flip is **statistically robust across seeds.**

> [!note] What this means for Phase 6
> Phase 6's Locke detector (E9500/E9501/E9502 in [[Cronos GAN Architecture]]) operates on Cronos GAN's regime_shift_score and eval_absolute_gap. Those signals being calibrated (not noise) is the load-bearing claim that justifies wiring them into a downstream data-validity belief score. Phase 4d's n_seeds=10 validation is what supplies that confidence.

The closest call — `SsmAsGen` vs `Symmetric` at 3% margin — is worth flagging. `Symmetric` is the "null hypothesis" (no challenger pressure at all), so a 3% improvement says the asymmetric pressure is *doing something productive but not dramatically so* over plain supervised training. The load-bearing comparison is `SsmAsGen` vs `LiqAsGen` (19% on eval_gap, 61% on eval_ssm_loss), which is unambiguous.

## Per-mode recommendation matrix

Given the Phase 4d findings, here's the practical recipe:

| Goal | Recommended mode | Rationale |
|---|---|---|
| "I want to surface calibrated regime-shift signals" | `SsmAsGenerator` | Headline finding — best held-out gap |
| "I want maximum training-MSE on the Liquid network" | `LiquidAsGenerator` (caveat: overfits) | Be aware the disagreement signal won't transfer |
| "I want a neutral baseline with no asymmetric coupling" | `Symmetric` | Useful as a sanity check / null hypothesis |
| "I want to compare all three" | `run_experiment_sweep` with `n_seeds ≥ 3` | The infrastructure that produced these tables |

## Reproducing the empirical flip

```bash
# In the cjc-cronos-gan crate root:
cargo test -p cjc-cronos-gan --release --test test_phase_4d -- --ignored

# Or run the single-seed visual demo:
cargo run --release -p cjc-cronos-gan --example sweep
```

The 4 `#[ignore]`'d tests take ~48 seconds total (10 seeds × 15 cells × eval). Each test prints its mean values via `eprintln!` regardless of pass/fail, so the actual numbers are visible in the terminal even when the assertion succeeds.

## Open empirical questions

The Phase 4d infrastructure supports more experiments that haven't been run yet:

1. **Decaying λ with `LiquidAsGenerator`**: does `LambdaSchedule::Linear { start: 0.15, end: 0.0, n_train_steps }` recover the training-MSE advantage while preserving generalization? The hypothesis is that early divergence pressure helps exploration; late no-pressure helps the network settle on a stable supervised solution. Untested.
2. **`WarmupThenLinear` for either asymmetric mode**: does a few epochs of pure supervised training before the challenger pressure kicks in produce qualitatively different findings? Untested.
3. **Per-mode `lr` sensitivity**: does `LiquidAsGenerator` benefit from a smaller `lr` to compensate for the noisier divergence gradient? Phase 4d's `per_mode_lr` field supports this; not yet exercised in a controlled experiment.
4. **Cross-dataset generalisation**: does training the GAN on `regime_shift` and evaluating on `step_change_anomaly` (transfer setting) preserve the empirical flip? Untested.

These are good Phase 7 starting points.

## See also

- [[Cronos GAN Phase 1 Overview]] — Phases 1-5 narrative including the original sweep
- [[Cronos GAN Architecture]] — pipeline + module map
- [[Cronos GAN Verification Report]] — the test suite that catches regressions
- [[Adversarial Temporal Training]] — the theory behind why SsmAsGen wins
