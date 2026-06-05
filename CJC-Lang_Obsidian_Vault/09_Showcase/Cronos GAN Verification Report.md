---
title: "Cronos GAN — Verification Report"
tags: [showcase, verification, tests, determinism, ci-matrix, bolero, fuzz, proptest]
status: "✅ 197 tests passing release + 5 ignored; CI matrix across 3 OSes"
crate: cjc-cronos-gan
version: 0.1.11
date: 2026-06-04
date-modified: 2026-06-04
---

# Cronos GAN — Verification Report

What gets tested, what guarantees those tests provide, and what the cross-platform CI matrix exercises on every PR. The companion to [[Cronos GAN Experiment Results]] — that doc says "what the model does", this one says "what we've verified about it."

## Test totals at Phase 6.2

**197 passing tests + 5 ignored** in release mode.

| Suite | Tests | Purpose | Wall clock (release) |
|---|---|---|---|
| `src/*` inline | 76 | Per-module unit tests + 14 schedule.rs tests | <1s |
| `tests/test_training.rs` | 12 | Phase 2: param round-trip + autograd vs FD + convergence + byte-id | ~0.5s |
| `tests/test_gan_training.rs` | 16 | Phase 3b: 3 modes × validation + roles + byte-id | ~2s |
| `tests/test_experiment_sweep.rs` | 21 | Phase 4b: 15-cell coverage + per-dataset lr + sweep_hash byte-id + mode-separation invariants | ~3s |
| `tests/test_phase_4c.rs` | 14 | Phase 4c: eval determinism + per-mode λ + Phase 3b invariants survive eval | ~2s |
| `tests/test_phase_4d.rs` | 21 + 4 ignored | Phase 4d: schedule math/integration + per-mode lr + multi-seed + format adaptation + 4 empirical-flip replication (ignored) | ~2s + 48s ignored |
| `tests/test_fuzz.rs` | 7 | Phase 5b: bolero fuzz × 1024 iters each | <1s |
| `tests/test_proptest.rs` | 10 | Phase 5: 64-256 cases per property | <1s |
| `tests/test_locke_detector.rs` | 19 | Phase 6.1: lift function + 3 detectors + bundle + malformed-DF safety | <1s |
| Doctest | 1 + 1 ignored | The lib.rs quick-start example | ~30s (cold) |

## Test pyramid

```
                ┌────────────────────────────┐
                │  4 empirical-flip tests    │   ← #[ignore]'d (slow but
                │   (Phase 4d, n_seeds=10)   │     load-bearing for Phase 6)
                ├────────────────────────────┤
                │  10 proptest properties    │   ← Phase 5: random inputs
                │   (64-256 cases each)      │
                ├────────────────────────────┤
                │  7 bolero fuzz targets     │   ← Phase 5b: adversarial bytes
                │   (1024 iterations each)   │     compiles to proptest in CI
                ├────────────────────────────┤
                │  136 integration tests     │   ← per-feature coverage
                ├────────────────────────────┤
                │  76 inline unit tests      │   ← per-module property checks
                └────────────────────────────┘
```

## The determinism contract — claim → test mapping

Every documented determinism claim has a test that would catch its violation. The 5 layers (from the [[Cronos GAN Architecture#Determinism stack|Architecture doc]]):

| Layer | Claim | Catching test |
|---|---|---|
| 1 — SplitMix64 substreams | Two domains with different salts produce different params | `ssm_and_liquid_use_independent_rng_substreams` (inline) |
| 1 — SplitMix64 substreams | Same seed twice ⇒ byte-identical params | `fuzz_random_seeds` (test_fuzz.rs) |
| 2 — Kahan accumulation | Reductions byte-identical across runs | `ssm_training_byte_identical_across_runs` (test_training.rs) |
| 3 — Adam moments | Training trajectory byte-identical across runs | `same_schedule_same_seed_byte_identical_trajectory` (test_phase_4d.rs) |
| 4 — Alternating updates | `(seed, config, inputs, targets)` ⇒ byte-identical TemporalGanTrainStep | `temporal_gan_byte_identical_across_runs` (inline) |
| 5 — replay_hash | Same `(ExperimentConfig, seed)` ⇒ same hash | `experiment_replay_hash_byte_identical_across_runs` (experiment.rs inline) |
| 5 — replay_hash | Schedule field changes ⇒ hash shifts | `changing_seed_stride_shifts_sweep_hash`, `changing_n_seeds_shifts_sweep_hash` (test_phase_4d.rs) |
| 6 — sweep_hash | `(SweepBaseConfig, seed, n_seeds, stride)` ⇒ same sweep_hash | `same_config_same_seed_same_stride_byte_identical_sweep_hash` (test_phase_4d.rs) |

## The cross-platform CI matrix

`.github/workflows/cjc-cronos-gan-determinism.yml` runs on every PR touching the crate. Triggered on push / PR to `master` / `main` when any of these paths change:

- `crates/cjc-cronos-gan/**`
- `tests/cronos/**` (anticipating the layout move from Phase 5b's deferred component)
- `Cargo.toml`, `Cargo.lock`
- The workflow YAML itself

### 6 gates × 3 OSes per PR

| Gate | Command | Purpose |
|---|---|---|
| 1 | `cargo test --release --lib` | 76 inline unit tests |
| 2 | `cargo test --release --test test_training` | 12 Phase 2 integration tests |
| 3 | `cargo test --release --test test_proptest` | 10 properties × 64-256 cases |
| 4 | `cargo test --release --doc` | Quick-start doctest |
| 5 | `cargo test --release --test test_phase_4d` | 21 Phase 4d integration tests |
| 6 | `cargo test --release --test test_fuzz` | 7 bolero targets × 1024 iters |

### Two byte-identity verification jobs

| Job | What it checks |
|---|---|
| `cronos-gan-byte-identity-check` | The Phase 5 proptest cases that check byte-identical rollout across runs. Three matrix runs must produce identical artifact values. |
| `cronos-gan-multi-seed-byte-identity` (Phase 5b new) | Two Phase 4d tests confirming sweep_hash is bit-identical across runs at `n_seeds=3`. Three matrix runs must agree. |

All jobs use `fail-fast: false`, so divergences on any one platform surface in the same CI run. `timeout-minutes` is 30 for the main test job and 15 for the byte-identity jobs.

## What each test catches — selected examples

### The fuzzer-found Liquid validation gap (Phase 5b)

The `fuzz_random_liquid_configs` bolero target found that `LiquidConfig` configurations with extreme `dt` (~1e234) + tiny `tau_min` (~1e-179) **pass** `validate()` (each field is individually finite-positive) but produce **NaN τ** during rollout — violating the documented "τ clipped by construction even under adversarial inputs" invariant.

The mechanism: `dt / tau_min` overflows f64 in the integration step. The current Phase 5b fuzz target uses bounded float ranges (`dt ∈ [0.01, 5]`, etc.) as a workaround. The deeper fix — tightening `LiquidConfig::validate` to reject overflow configs — is a spawned follow-up task.

This is the headline example of "fuzz finds bug": the invariant was conditionally true, the conditions weren't enforced, and the harness caught it within seconds of being added.

### The Phase 3b structural invariant (test_experiment_sweep.rs)

`ssm_loss_in_ssm_as_generator_equals_ssm_loss_in_symmetric` exercises the claim that the SSM trains identically as predictor in both `Symmetric` and `SsmAsGenerator` modes. If the asymmetric loss term ever accidentally leaks into the predictor's gradient (a structural bug in `TemporalGanTrainer`), this test would catch it across all 5 datasets simultaneously.

The mirror test `liquid_loss_in_liquid_as_generator_equals_liquid_loss_in_symmetric` is the symmetric guard for the other asymmetric mode.

### The empirical-flip replication (Phase 4d, #[ignore]'d)

The 4 `#[ignore]`'d Phase 4d tests assert the Phase 4c flip holds at `n_seeds=10`. They're load-bearing for Phase 6's Locke detector — if the flip is unstable across seeds, the detector is reporting noise, and Phase 6 should not have been built on top of it.

Why ignored: 48s total in release. Re-run with `cargo test --release --test test_phase_4d -- --ignored` whenever the trainer or the schedule logic changes. The 3 inequalities all hold with strict `<` (61%, 19%, 3% margins) — see [[Cronos GAN Experiment Results]] for the actual numbers.

### Detector silent-no-op on malformed DataFrames (Phase 6.1)

`detectors_silently_no_op_on_malformed_dataframe` (test_locke_detector.rs) feeds a DataFrame with no Cronos columns into all 3 detectors. They must emit zero findings and never panic.

This protects against the integration pattern where a user attaches `cronos_default_detectors()` to a `validate(df, opts)` call against a non-Cronos DataFrame — a real footgun if the detectors crashed instead of no-op'ing.

## The fuzz-target adversarial contract

The 7 Phase 5b bolero targets each enforce one or more of these contracts on **every** random byte sequence:

1. **No panic** — the public API treats malformed configs as user errors and returns `CronosGanError`. A panic on any random input is a bug.
2. **No silent NaN propagation** — when a network ingests non-finite input, it must either return an explicit `NonFiniteInput` error OR a fully-finite output. NaN sneaking through is a bug.
3. **Byte-identical replay** — same `(config, seed, inputs)` ⇒ same parameter bit pattern, regardless of how exotic the seed value is.
4. **Structural invariants survive bounded inputs** — `‖A‖_F ≤ α` for the SSM, `τ ∈ [tau_min, tau_max]` for the Liquid (per the documented bound; see the LiquidConfig validation gap above for the conditional caveat).

The fuzz targets compile to proptest under `cargo test` (1024 iters each in CI) and route to libfuzzer / AFL under `cargo bolero` for coverage-guided mutation. Adding a target costs <100 LOC; the per-CI cost is ~50ms per target.

## What we explicitly DON'T test

To set expectations clearly:

- **Forecasting accuracy vs. statsmodels / Prophet / N-BEATS** — see the Cargo.toml docstring caveat. The crate is correctness-first, not accuracy-first.
- **Long-range modeling capacity** — `state_dim` is typically ≤ 64. We don't claim to compete with S4 or Mamba on Long Range Arena.
- **Bit-identity across hardware acceleration backends** — we have no GPU/SIMD path. The byte-identity claim is for the scalar CPU code path only.
- **Cross-version replay** — within a salt-version (e.g. `v4d`), replay is byte-identical. Across salt-versions (`v4c` → `v4d`), the hashes shift by design. There's no "old hash compatibility" claim.
- **Adversarial security** — the bolero fuzz targets find correctness bugs, not security bugs. The crate has no security boundary; the `CronosGanError`-not-panic contract is about helpfulness to honest users, not exploit prevention.

## Open verification items

- The LiquidConfig validation gap fix (spawned task) + corresponding test exercising the formerly-accepted-now-rejected config.
- The cross-platform CI matrix has been written but hasn't yet fired on a real PR (the branch is local-only at time of writing).
- A dedicated test for the Phase 6.1 Locke detector composition with `cjc_locke::validate(df, opts)` — currently the detectors are tested in isolation via direct `FindingSink::new` + `detector.run(df, &mut sink)`. A full integration test through `validate()` would also exercise the detector-ordering + finding-merge contract.

## See also

- [[Cronos GAN Architecture]] — pipeline + module map
- [[Cronos GAN Experiment Results]] — the findings these tests gate
- [[Adversarial Temporal Training]] — the structural-invariant theory
- [[State Space Model Primitive]] / [[Liquid Neural Network Primitive]] — per-primitive invariants
