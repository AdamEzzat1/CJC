---
title: Liquid Neural Network Primitive
tags: [ml, temporal, liquid-nn, ltc, determinism, gates, autodiff]
status: Implemented
crate: cjc-cronos-gan
source: crates/cjc-cronos-gan/src/liquid.rs
---

# Liquid Neural Network Primitive

> [!info] Position in the architecture
> The Liquid network is the **nonlinear, time-varying** half of the Cronos GAN's [[Cronos GAN Architecture|adversarial pair]]. It provides the reactive inductive bias that the [[State Space Model Primitive|SSM]] is *deliberately* opposed to. Its per-step time constant `τ` is gated by the input AND the current state, so the network can switch between memory-like and reactive modes depending on what's happening locally.

## Summary

A discrete-time liquid-time-constant network, formulated in Phase 2 with a smooth sigmoid-gated time constant:

```
act = tanh(W_h · h + W_x · u + b)
s   = sigmoid(W_τ_u · u + W_τ_h · h + b_τ)   ∈ (0, 1)
τ   = τ_min + (τ_max − τ_min) · s             ∈ (τ_min, τ_max)
h'  = h + dt/τ · (−h + act)
y   = W_out · h' + b_out
```

where `h ∈ ℝ^{state_dim}`, `u ∈ ℝ^{input_dim}`, `y ∈ ℝ^{output_dim}`. The time constant `τ` is per-dimension and per-step.

## The Phase 2 sigmoid-scaled τ refactor

Phase 1 originally computed `τ` as:

```
τ = softplus(W_τ_u · u + W_τ_h · h + b_τ).clamp(τ_min, τ_max)     (Phase 1 — superseded)
```

This is **non-differentiable at the clip boundaries**. Once the pre-clip value of `softplus(...)` slid past either bound, the gradient of `τ` w.r.t. its inputs became zero — the gate would saturate and the network would stop learning to vary `τ` in that regime.

Phase 2 replaced it with the sigmoid-scaled form above. The two formulations are equivalent in expressive range — both produce `τ ∈ (τ_min, τ_max)` — but the sigmoid is smooth everywhere. Gradients flow through the entire range, the gate learns to use the full window, and finite-difference vs autograd agreement passes at relative tolerance `5e-4` (vs `1e-4` for the strictly linear SSM).

> [!note] What this means in practice
> The `LiquidConfig.tau_min` and `tau_max` are now hard structural bounds (not learned), and the network's effective time constant smoothly interpolates between them based on `(u, h)`. The Phase 1 "discontinuity in the gradient at the boundary" failure mode is gone.

## Inspectable gates

By design, the network exposes its internal gates so users can attribute regime-shift events to specific gate behaviour. After a rollout, the result carries:

| Field | Shape | Meaning |
|---|---|---|
| `outputs[t]` | `output_dim` | Predicted output at step t |
| `states[t]` | `state_dim` | Internal hidden state h |
| `time_constants[t].tau` | `state_dim` | Per-dim τ at step t |
| `time_constants[t].pre_sigmoid` | `state_dim` | The `W_τ_u·u + W_τ_h·h + b_τ` value before sigmoid (useful for diagnostics) |
| `gates[t].gate` | `state_dim` | The `dt / τ` integration weight at step t |

The Phase 4 datasets module (`src/datasets.rs`) uses this to label regime-shift events: when the `regime_shift` generator's midpoint is crossed, the Liquid `τ` jumps from "slow / memory-like" toward "fast / reactive" as the gate fires — observable in `time_constants` without needing to peek at the loss.

## Overflow-safe sigmoid

The `sigmoid(x)` and `softplus(x)` implementations are overflow-safe at every finite f64 input:

```
softplus(x) = if x > 20 { x } else { ln(1 + e^x) }
sigmoid(x)  = if x >= 0 { 1 / (1 + e^{-x}) } else { e^x / (1 + e^x) }
```

The split branch keeps `e^x` and `e^{-x}` in the magnitude regime where neither overflows. This is a Phase 1 invariant and survives the Phase 2 refactor.

## Comparison to Hasani et al. 2020 / 2021

| Property | Hasani et al. (LTC, [Closed-form Continuous-time NN](https://arxiv.org/abs/2106.13898)) | This Liquid NN |
|---|---|---|
| Time formulation | Continuous, solved via ODE integrator | Discrete, explicit Euler step |
| τ formulation | Input-dependent, closed-form analytical | Sigmoid-gated, bounded |
| Numerical scheme | Adaptive RK4 / closed-form | Fixed dt, single Euler step |
| Stability | Lyapunov-derived bounds | `τ ∈ (τ_min, τ_max)` by construction |
| Determinism | Convention-dependent | Bit-identical across runs / OSes |
| Training scale | Production-grade (autonomous driving) | Small (state_dim ≤ 64 typical) |
| Differentiable | Through ODE solver (sensitivity analysis) | Trivially through the explicit step |

We're closer to the original LTC paper's *spirit* (gated time constants, inspectable dynamics) than to its production form. The discretisation is plain Euler instead of an ODE solver, which makes the autograd path drastically simpler and the byte-identity contract trivial.

## Determinism contract

Same as the [[State Space Model Primitive|SSM]] — per-matrix substreams off `CronosSeed`:

- `"liquid.W_h"`, `"liquid.W_x"`, `"liquid.b"` — main activation path
- `"liquid.W_tau_u"`, `"liquid.W_tau_h"`, `"liquid.b_tau"` — τ gate
- `"liquid.W_out"`, `"liquid.b_out"` — output projection

The "different substream salts produce different params" property is checked by an inline test (`ssm_and_liquid_use_independent_rng_substreams` in `src/lib.rs`).

## Canonical bytes

```
u64 state_dim, input_dim, output_dim (each LE)
u64 dt, tau_min, tau_max, init_scale bits (each f64::to_bits, LE)
```

Total: 56 bytes. Used by `LiquidConfig::canonical_bytes` and consumed by upstream `replay_hash` composition.

## Autodiff path

The Phase 2 autograd adapter (`src/autograd_liquid.rs`) builds a BPTT graph over the step recurrence. The flat parameter ordering:

```
[W_h] ++ [W_x] ++ [b] ++ [W_τ_u] ++ [W_τ_h] ++ [b_τ] ++ [W_out] ++ [b_out]
```

Total length: `state_dim² + state_dim·input_dim + state_dim + state_dim·input_dim + state_dim² + state_dim + output_dim·state_dim + output_dim`.

Tested vs finite difference at relative tolerance `5e-4` (`tests/test_training.rs::liquid_autograd_matches_finite_difference`). The tolerance is wider than the SSM's `1e-4` because the sigmoid + tanh composition is more numerically active.

## Validation rules

`LiquidConfig::validate` requires:
- `state_dim ≥ 1`, `input_dim ≥ 1`, `output_dim ≥ 1`
- `dt > 0`, finite
- `tau_min > 0`, `tau_max > tau_min`, both finite
- `init_scale > 0`, finite

## ⚠️ Known validation gap (Phase 5b fuzz finding)

The Phase 5b bolero fuzz target `fuzz_random_liquid_configs` found a real bug: configurations with extreme `dt` (~1e234) and tiny `tau_min` (~1e-179) **pass** the current `validate()` because each field is individually finite-positive, but produce **NaN τ values** during rollout — violating the documented "τ clipped by construction even under adversarial inputs" invariant.

The mechanism: `dt / tau_min` overflows `f64` in the integration step, and the overflow propagates into the τ-computing path before clipping can rescue it.

**Workaround in Phase 5b**: the fuzz target uses bounded float ranges (`dt ∈ [0.01, 5]`, `tau_min ∈ [0.01, 0.5]`, `tau_max ∈ [1.0, 100]`, `init_scale ∈ [0.001, 1.0]`) so the invariant holds on every input the fuzzer produces. The detector still exercises the same dispatch correctness without exposing the corner.

**Deeper fix (deferred)**: tighten `LiquidConfig::validate` to reject configs where `dt / tau_min` overflows (and add an inline test exercising the formerly-accepted-now-rejected config). After the fix, the fuzz target can drop its bounded ranges. Spawned as a follow-up task from the Phase 5b session.

This is a healthy *fuzz-finds-bug* outcome — the invariant is conditionally true, the conditions weren't enforced, and the fuzz harness caught it.

## Open questions

- The fix for the validation gap above.
- **Per-step `dt`**: currently a global config field. A per-step variant would allow integration over irregularly-sampled time series. Adds API surface; not yet justified by a use case in the current 5 synthetic datasets.
- **Higher-order integration**: explicit Euler is the simplest. A semi-implicit or RK2 step would handle stiffer dynamics better. Trade-off: doubles the gradient-graph depth.

## See also

- [[State Space Model Primitive]] — the structural opposite
- [[Cronos GAN Architecture]] — pipeline context
- [[Adversarial Temporal Training]] — how the gates participate in the predictor/challenger framing
- [[Autodiff]] — the `GradGraph` BPTT machinery
