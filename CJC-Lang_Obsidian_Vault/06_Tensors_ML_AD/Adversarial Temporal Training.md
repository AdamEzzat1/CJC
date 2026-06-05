---
title: Adversarial Temporal Training
tags: [ml, temporal, gan, ncl, negative-correlation-learning, predictor-challenger, determinism]
status: Implemented
crate: cjc-cronos-gan
source: crates/cjc-cronos-gan/src/gan_trainer.rs, crates/cjc-cronos-gan/src/training.rs
---

# Adversarial Temporal Training

The training loop that opposes the [[State Space Model Primitive|SSM]] and [[Liquid Neural Network Primitive|Liquid network]] without losing supervision against the target. Ships in Phase 3b; extended in Phase 4d with [`LambdaSchedule`].

> [!info] Companion notes
> See [[Cronos GAN Architecture]] for the full pipeline view and [[Cronos GAN Experiment Results]] for the empirical findings that this training procedure produces.

## The framing question

Standard GANs train a generator to fool a discriminator. Cronos GAN's brief is fundamentally different: **disagreement is the artifact**, not the loss to minimise. If you train one network to produce data the other can't distinguish from real, you end up with predictions that look real. That's the wrong output here — we want predictions that *both networks are reasonably accurate on* but that *disagree on uncertain regions* in a calibrated way.

This requires two changes from the standard GAN setup:
1. Both networks see the **same** input and target.
2. The asymmetric loss term *rewards* divergence, but is bounded by the supervised MSE that keeps both networks honest.

## The three modes

| Mode | SSM role | Liquid role | What the asymmetric term does |
|---|---|---|---|
| `Symmetric` | Predictor | Predictor | No asymmetric term — both pure MSE |
| `SsmAsGenerator` | Predictor | Challenger | Liquid rewarded for diverging from SSM's prediction |
| `LiquidAsGenerator` | Challenger | Predictor | SSM rewarded for diverging from Liquid's prediction |

(The names `Predictor` / `Challenger` are the cjc-cronos-gan-specific labels for the GAN literature's `Generator` / `Discriminator`, chosen because the contest isn't fool-vs-detect.)

## The loss

For the predictor `p` and challenger `c`, given inputs `u`, target `y`:

```
L_predictor  =  MSE(p(u), y)                              (vanilla supervised)
L_challenger =  MSE(c(u), y)  −  λ · MSE(c(u), p(u))      (supervised − divergence)
```

The negative sign is the entire asymmetric trick: it rewards the challenger for finding solutions that are **accurate against the target** but **different from the predictor's solution**. Both terms are MSEs, both bounded below by zero, both differentiable.

## NCL — Negative Correlation Learning

This is exactly the loss function from [Liu & Yao 1999, "Ensemble Learning via Negative Correlation"](https://www.sciencedirect.com/science/article/pii/S0893608099000738). For an ensemble of `M` learners `{f_i}`, NCL trains each one with:

```
L_i = ½ (f_i − y)²  −  λ · (f_i − f̄) · Σ_{j≠i} (f_j − f̄)
```

where `f̄` is the ensemble mean. For `M = 2` and `f̄ = (f_1 + f_2)/2`:

```
L_1 = ½ (f_1 − y)²  −  λ · (f_1 − f̄) · (f_2 − f̄)
    = ½ (f_1 − y)²  −  λ · (f_1 − (f_1+f_2)/2) · (f_2 − (f_1+f_2)/2)
    = ½ (f_1 − y)²  −  λ · ((f_1 − f_2)/2) · (−(f_1 − f_2)/2)
    = ½ (f_1 − y)²  +  (λ/4) · (f_1 − f_2)²
```

The challenger loss `MSE(c, y) − λ · MSE(c, p)` is the same shape (modulo the sign convention that determines whether you want correlation or anti-correlation, and the scale factor `1/4` vs. `1` which is absorbed into the choice of `λ`).

This is genuinely *known* literature — what's new in Cronos GAN is the **temporal** application (NCL has historically been applied to ensemble *predictions*, not to temporal *trajectories*) and the **structural opposition** (Liu & Yao's ensemble was homogeneous; here the two networks have intentionally different inductive biases).

## Predictor-first ordering

Per `TemporalGanTrainer::step`:

```
1. Compute predictor's MSE loss against target, run backward + Adam update.
2. Forward the *updated* predictor to get its new per-step outputs.
3. Build ChallengerSpec with those outputs + the current λ.
4. Compute challenger's loss (supervised MSE − λ · MSE-vs-predictor),
   backward + Adam update.
5. Increment step_count.
```

The ordering matters for **determinism**:

- The predictor's update is purely a function of `(predictor state, inputs, targets)`. Same inputs ⇒ same predictor update.
- The challenger reads the *post-update* predictor outputs as a fixed reference. Since the predictor update was deterministic, those outputs are deterministic too.
- The challenger's update is then a function of `(challenger state, inputs, targets, predictor outputs, λ)`. All deterministic.

If we reversed the ordering (challenger first, then predictor), the challenger would read the *previous step's* predictor outputs, which is also deterministic but introduces a one-step lag in the divergence signal. The current ordering keeps the disagreement maximally fresh.

## The determinism contract

> **Same `(seed, config, inputs, targets, initial Adam state)` ⇒ byte-identical predictor update ⇒ byte-identical predictor outputs ⇒ byte-identical challenger update ⇒ byte-identical `TemporalGanTrainStep`.**

Phase 3b's tests `tests/test_gan_training.rs` exercise this across all three modes (Symmetric, SsmAsGenerator, LiquidAsGenerator). The cross-platform CI matrix (Phase 5 + 5b, see [[Cronos GAN Verification Report]]) re-runs the byte-identity assertions on Ubuntu + Windows + macOS to verify the contract holds across OSes.

## The Phase 3b structural-test invariants

Two invariants follow from the framing and are exercised explicitly:

### Invariant A (SSM-predictor invariant)

> In `Symmetric` mode, the SSM trains identically as predictor.
> In `SsmAsGenerator` mode, the SSM is ALSO the predictor.
> Therefore: with the same seed, the SSM's final loss is byte-identical between these two modes.

Test: `ssm_loss_in_ssm_as_generator_equals_ssm_loss_in_symmetric` in `tests/test_experiment_sweep.rs`. Across all 5 datasets:

```
sym.final_loss_ssm.to_bits() == ssm_as_gen.final_loss_ssm.to_bits()    ∀ dataset
```

This is a property of the training procedure: the asymmetric term only affects the *challenger*, so the predictor's trajectory is preserved.

### Invariant B (Liquid-predictor invariant)

Mirror image of A. In `LiquidAsGenerator`, the Liquid is the predictor; its final loss equals the Symmetric Liquid loss:

```
sym.final_loss_liquid.to_bits() == liq_as_gen.final_loss_liquid.to_bits()  ∀ dataset
```

Test: `liquid_loss_in_liquid_as_generator_equals_liquid_loss_in_symmetric`.

Phase 4c added the eval-pass versions:

```
sym.eval.ssm_loss.to_bits() == ssm_as_gen.eval.ssm_loss.to_bits()      ∀ dataset
sym.eval.liquid_loss.to_bits() == liq_as_gen.eval.liquid_loss.to_bits()  ∀ dataset
```

If either invariant ever fails, the asymmetric loss is leaking into the predictor's update — a structural bug in the trainer.

## Phase 4d — LambdaSchedule

Pre-Phase-4d, `λ` was a flat `f64` carried on `TemporalGanConfig`. Phase 4d generalised it to [`LambdaSchedule`] (four variants: `Constant`, `Linear`, `ExponentialDecay`, `WarmupThenLinear`). Each non-constant variant carries its own `n_train_steps` horizon, so `schedule.lambda_at(step)` is fully self-contained — the trainer just feeds in its own `step_count`.

The motivation: the Phase 4c finding that `SsmAsGenerator` wins held-out evaluation suggested that the "ideal λ" might not be constant — early training benefits from larger divergence pressure (encourage exploration), while late training benefits from smaller pressure (settle on a stable disagreement pattern). The schedule machinery lets us test that hypothesis without changing the trainer's call sites.

Schedule semantics:

| Variant | Formula | Use case |
|---|---|---|
| `Constant(v)` | `v` at every step | Default; matches pre-Phase-4d behaviour |
| `Linear { start, end, n_train_steps }` | Linear interp `[start → end]` over `[0, n_train_steps]`; clamped to `end` past | Regularisation warmup that decays into vanilla supervised |
| `ExponentialDecay { start, decay_rate, n_train_steps }` | `start · exp(−decay_rate · t / n_train_steps)` | Smooth monotonic decay; never reaches 0 |
| `WarmupThenLinear { start, end, warmup_steps, n_train_steps }` | `start` during warmup, then linear to `end` | Let predictor stabilise before challenger pressure ramps up |

The schedule's `canonical_bytes()` includes a per-variant tag byte plus all fields, so two schedules with the same tag but different fields produce different bytes — the replay-hash invariant survives the new shape.

## Per-mode lr

Phase 4d also added per-mode learning-rate overrides to `SweepBaseConfig`:

```
lr resolution: per_dataset_lr[dataset] → per_mode_lr[mode] → default_lr
```

Most-specific wins. Useful when the asymmetric modes benefit from a slower lr (the challenger's gradient has a divergence term that can be noisier).

## Multi-seed Welford for sample variance

Phase 4d's `SweepCell` carries `mean: CellAggregate` and `variance: CellAggregate`, computed via Kahan-summed mean and Welford's online sample-variance recurrence:

```
M_1 = x_1, S_1 = 0
for k = 2 to n:
    M_k = M_{k-1} + (x_k - M_{k-1}) / k
    S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
variance_sample = S_n / (n - 1)
```

For `n_seeds = 1`, sample variance is undefined; we use `0.0` by convention to keep downstream `sqrt(variance)` finite. The Phase 4d aggregation is what enabled the headline empirical validation in [[Cronos GAN Experiment Results]] (the n_seeds=10 confirmation of the Phase 4c flip).

## Open questions

- **Decaying-λ recipe**: the Phase 4c open question — does `Linear { start: 0.15, end: 0.0, n_train_steps }` for `LiquidAsGenerator` recover the training-MSE advantage while preserving generalization? The infrastructure ships in Phase 4d; the empirical answer awaits a dedicated sweep.
- **Beyond pairs**: NCL generalises to `M ≥ 3` networks. A trio of opposing inductive biases (e.g. SSM + Liquid + a Mamba-style selective SSM) could in principle produce a richer disagreement signal. Out of current scope.
- **The "soft inverse" framing**: the predictor/challenger framing has a parallel in inverse problems — the challenger is approximating the *posterior variance* of the supervised problem. Formalising this connection is a vault note that hasn't been written yet.

## See also

- [[State Space Model Primitive]] / [[Liquid Neural Network Primitive]]
- [[Cronos GAN Architecture]]
- [[Cronos GAN Experiment Results]] — the empirical-flip discovery that validates the framing
- [[Autodiff]] — the BPTT machinery the trainer uses
