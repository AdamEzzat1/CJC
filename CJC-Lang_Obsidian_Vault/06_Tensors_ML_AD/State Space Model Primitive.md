---
title: State Space Model Primitive
tags: [ml, temporal, ssm, determinism, structural-stability]
status: Implemented
crate: cjc-cronos-gan
source: crates/cjc-cronos-gan/src/ssm.rs
---

# State Space Model Primitive

> [!info] Position in the architecture
> The SSM is one half of the Cronos GAN's [[Cronos GAN Architecture|adversarial pair]]. It provides the **stable, linear, time-invariant** inductive bias that the [[Liquid Neural Network Primitive|Liquid network]] is *deliberately* opposed to. Together they generate the calibrated disagreement that Cronos GAN's `TemporalDisagreement` measures.

## Summary

A deterministic, noise-free, discrete-time linear state-space model:

```
x_{t+1} = A · x_t + B · u_t
y_t     = C · x_t + D · u_t       (D fixed to 0 in Phase 1)
```

where `x ∈ ℝ^{state_dim}`, `u ∈ ℝ^{input_dim}`, `y ∈ ℝ^{output_dim}`. Matrices `A, B, C` are learnable; `D` is fixed at zero.

The construction guarantees **`‖A‖₂ ≤ α < 1`** by construction — stability is a property of the matrix initialisation, not of training. With `α = 0.95` (the default), the state's transient decay is bounded above by `α^t = 0.95^t`, and no rollout can blow up under any finite input.

## Structural-stability argument

The transition matrix `A` is constructed as:

```
A = (α / √state_dim) · R
```

where each row of `R ∈ ℝ^{state_dim × state_dim}` is drawn from a standard normal, then **L²-normalised to unit length**.

The Frobenius norm of a row-normalised matrix is exactly `√state_dim`, so:

```
‖A‖_F = (α / √state_dim) · √state_dim = α     (exactly)
```

And since `‖A‖₂ ≤ ‖A‖_F` for any matrix, `‖A‖₂ ≤ α < 1`.

This is what we mean by *structural* stability: every spectral norm bound holds as a property of the construction, with no slack and no dependence on the training dynamics. The Phase 5b fuzz target `fuzz_random_ssm_configs` exercises this invariant directly over a wide range of `(state_dim, alpha, init_scale)` triples.

## Why linear + time-invariant

The SSM is deliberately chosen as the *simpler* half of the pair:

- **Linear in (x, u)** — every gradient exists trivially. The autograd path through SSM is the simplest in the workspace.
- **Time-invariant** — `A, B, C` don't depend on the timestep. The decay constant `α` is the only temporal scale; the model has no opinion about volatility or local nonlinearity.
- **Stable** — guaranteed `‖A‖₂ < 1` means any rollout converges to a fixed point for stationary inputs.

The point is to opt OUT of all the flexibility a learner might want, and instead provide a clean baseline that exposes the rich behaviour of the [[Liquid Neural Network Primitive|Liquid network]] by contrast. When the SSM and the Liquid network disagree, the disagreement is the Liquid network's nonlinearity speaking — not noise from a flexible-but-erratic SSM.

## Comparison to S4 / Mamba

We're not in the same regime as the [S4](https://arxiv.org/abs/2111.00396) or [Mamba](https://arxiv.org/abs/2312.00752) state-space models, and we don't claim to be:

| Property | S4 / Mamba | This SSM |
|---|---|---|
| Goal | Long-range sequence modeling competitive with Transformers | Stable baseline for adversarial pairing |
| Discretisation | HiPPO + bilinear / zero-order hold | Plain discrete-time |
| Parameterisation | Diagonal-plus-low-rank, structured | Dense, row-normalised |
| Selectivity | Input-dependent (Mamba) | Time-invariant |
| Stability | Stable by parametrisation choices | Stable by `‖A‖_F = α` construction |
| Training scale | Billions of params | `state_dim × (state_dim + input_dim + output_dim)` (tens to thousands) |
| Determinism | Convention-dependent | Bit-identical replay across runs / OSes |

The Cronos GAN SSM trades modeling capacity for **structural guarantees and replay**. It will not beat a tuned S4 on Long Range Arena; that is not the contest.

## Determinism contract

Every random draw routes through [`CronosSeed::substream`] with a per-matrix salt:

- `"ssm.A"` — the rows of `R` before normalisation
- `"ssm.B"` — the input matrix
- `"ssm.C"` — the output matrix

Two distinct salts can never share an RNG stream — the substream-derivation function takes the salt bytes into the SplitMix64 jump, so each matrix's noise is independent by construction.

All reductions (matrix-vector products, row L² norms, loss sums) route through [`cjc_repro::KahanAccumulatorF64`]. The Box-Muller standard-normal draws consume exactly two uniforms per draw, matching the cjc-tempest sampler convention so cross-stream determinism debugging is uniform across the workspace.

## Canonical bytes

The hash-stable byte layout used for `replay_hash` composition (see `StateSpaceConfig::canonical_bytes`):

```
u64 state_dim (LE)
u64 input_dim (LE)
u64 output_dim (LE)
u64 alpha bits (f64::to_bits)
u64 init_scale bits
```

Any change to any field changes the hash. This is the contract that `cronos_experiment_replay_hash_v4d` consumes upstream.

## Autodiff path

The Phase 2 autodiff adapter (`src/autograd_ssm.rs`) lifts the SSM into a `cjc_ad::GradGraph` BPTT computation. The flat parameter ordering is:

```
[A row-major] ++ [B row-major] ++ [C row-major]
```

Length: `state_dim² + state_dim·input_dim + output_dim·state_dim` (D is fixed at zero, not a parameter).

The `Trainable` trait methods `params_flat`, `set_params_flat`, `build_rollout_graph` are tested against finite-difference gradients with relative tolerance `1e-4` in `tests/test_training.rs::ssm_autograd_matches_finite_difference`.

## Validation rules

`StateSpaceConfig::validate` (called by `from_seed`) requires:
- `state_dim ≥ 1`
- `input_dim ≥ 1`
- `output_dim ≥ 1`
- `0 < alpha < 1`, finite
- `init_scale > 0`, finite

The Phase 5b fuzz target [`fuzz_random_ssm_configs`](source:crates/cjc-cronos-gan/tests/test_fuzz.rs) verifies that *every* config either passes validation and produces a finite model with `‖A‖_F ≤ alpha`, or fails with `InvalidConfig`. No third option, no panic.

## Open questions

- **Extreme `alpha`** (`< 1e-100` or close to `1.0 - ε`): currently accepted by `validate`. The matrix-construction math handles these gracefully but the model's behaviour at the boundary may not match the user's intent. A follow-up tightening could clamp alpha to `[1e-3, 0.999]`.
- **`D ≠ 0` direct feedthrough**: not parameterised in Phase 1. Adding it is a Phase 7+ scope decision since it changes the autograd path and the canonical-bytes layout (requires a salt bump).
- **Continuous-time SSM**: the discrete-time form was the simplest deterministic baseline. A continuous-time variant (and the corresponding discretisation choice) would be a separate primitive, not a modification of this one.

## See also

- [[Cronos GAN Architecture]] — where this primitive sits in the pipeline
- [[Liquid Neural Network Primitive]] — the structural opposite
- [[Adversarial Temporal Training]] — how the two are trained against each other
- [[Autodiff]] — the `GradGraph` infrastructure backing `autograd_ssm`
- [[ADR-0004]] (SplitMix64 RNG) — the substream-derivation contract
