# cjc-causal — formal causal inference for CJC-Lang

Deterministic, content-addressed treatment-effect estimates. Same data, same
seed, same assumptions ⇒ byte-identical `EffectEstimate.identifier`. Across
runs, across platforms, across library upgrades.

## Why cjc-causal

Causal-inference incumbents in Python — DoWhy, EconML, CausalML — produce
estimates whose stochastic components (bootstrap CIs, cross-fitting fold
assignment, propensity-score matching tie-breaks) shift between runs because
the libraries derive their RNG from system entropy or rely on hash-order
iteration.

cjc-causal threads `(seed, data, assumptions)` through every stochastic step
so a published `EffectEstimate.identifier` is a content-addressed fingerprint
anyone can re-derive. The hash is over the canonical byte representation of
the inputs and the estimate; matching identifier ⇒ matching estimate, ULP for
ULP, no tolerance window required.

This matters specifically for:

- **Audit logs** — regulators asking "show me the analysis that produced this
  claim" get a hash that resolves to one identifiable computation.
- **Sensitivity analysis** — two runs differing only in a single assumption
  produce two distinct identifiers; the diff is the sensitivity.
- **Pre-registration** — pre-committing to a hash before seeing data, then
  publishing the matching hash on the analysis, removes the "you ran 47
  specifications and reported the best one" critique.

## Quick start

```rust
use cjc_causal::{IdentificationAssumption, PropensityScoreMatcher};

// `df` is a cjc_data::DataFrame; `locke_report` comes from a prior
// `cjc_locke::api::validate(&df, &cfg)` call.
let matcher = PropensityScoreMatcher::new()
    .with_caliper_sd(0.2)   // Austin 2011 default
    .with_seed(42);

let estimate = matcher.estimate(
    &df,
    "treatment",
    "outcome",
    &["age", "income", "education"],
    &[
        IdentificationAssumption::Unconfoundedness,
        IdentificationAssumption::Positivity,
        IdentificationAssumption::NoInterference,
    ],
    &locke_report,
)?;

println!(
    "ATT: {} ± {} (id: {})",
    estimate.point, estimate.std_error, estimate.identifier
);
# Ok::<(), cjc_causal::CausalError>(())
```

The `identifier` is a `cjc_locke::id::FingerprintId` (64-bit content hash).
Two runs of the snippet above with the same `df` produce the same identifier;
running again on perturbed data produces a different identifier.

## v0.1 estimator surface

- **`PropensityScoreMatcher`** — IRLS logistic regression for propensity
  scores, then greedy nearest-neighbour matching on the logit with an
  Austin-2011 caliper (`0.2 × SD(logit)`). Ties resolve by ascending row
  index, never by hash order. ATT via Kahan-summed paired differences.
  Pair-level bootstrap CI from `cjc_repro::Rng`.

- **`IVRegression`** — Just-identified two-stage least squares (2SLS) with
  Stock-Yogo weak-instrument F-statistic surfaced as a Locke `E9100` finding
  when F < 10. HC1 sandwich standard errors:
  `(n/(n-k)) · (X'X)⁻¹ · X'·diag(e²)·X · (X'X)⁻¹`. Confidence intervals via
  Acklam 2003 normal quantile.

- **`DoubleMLEstimator`** — Chernozhukov et al. 2018 partial-linear DML with
  K-fold cross-fitting (default K = 5). Linear nuisance functions in v0.1
  (via `cjc_runtime::hypothesis::lm`); MLP nuisances via `cjc_ad::GradGraph`
  deferred to v0.2. Orthogonal moment estimator
  `β̂ = Σ(T - t̂)(Y - ŷ) / Σ(T - t̂)²` with plug-in variance.

All three return [`EffectEstimate`] with `point`, `std_error`, optional
confidence interval, the user's declared `assumptions`, the IV first-stage
F-statistic (when applicable), and the content-addressed `identifier`.

## Determinism contract

1. All floating-point reductions go through `cjc_repro::KahanAccumulatorF64`.
   Raw `Vec<f64>::iter().sum()` is banned anywhere in the crate.
2. All map iteration uses `BTreeMap` / `BTreeSet`. No `HashMap`.
3. All randomness routes through `cjc_repro::Rng` (SplitMix64) with the seed
   threaded explicitly from the caller.
4. Matching ties resolve by ascending row index via strict-less-than
   comparison.
5. No FMA. `RUSTFLAGS` must not enable `target-feature=+fma`.

See ADR-0043 §"Determinism contract" for the full seed-flow diagram and the
proof that the `EffectEstimate.identifier` byte representation does not
depend on iteration order anywhere.

## Composing with cjc-locke

Every estimator takes a `&cjc_locke::LockeReport`. If the report contains
fatal findings on the analysis inputs — `E9001` missingness ≥ 30% on
treatment or outcome, `E9009` missing-but-promotable on continuous covariates,
`E9060` target-leakage on covariates — the estimator returns
`CausalError::DataQualityRefusal { findings }` immediately. The user gets
the offending findings attached so they can fix-or-acknowledge before
forcing the estimate.

This is a deliberate refusal pattern: cjc-causal will not produce an
identifier for an estimate the upstream data couldn't support. The Locke
report is the contract.

## What's deferred to v0.2

- **Over-identified 2SLS** (multiple instruments) with overidentification
  test. v0.1 ships just-identified 2SLS only.
- **MLP nuisances** for `DoubleMLEstimator` via `cjc_ad::GradGraph` and
  `cjc_runtime::adam_step`. Cleanly extends via a `NuisanceMode` builder
  argument; primitives are already in the workspace.
- **Mediation analysis**, **regression discontinuity**,
  **difference-in-differences**, **synthetic control**, **causal forests**.
- **do-calculus** + **structural equation modelling** — out of scope for
  v0.1; the handoff §2.7 carries the full deferral list.

## See also

- ADR-0043 design doc:
  [`CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0043 cjc-causal v0.1 — Propensity Score, IV, Double ML.md`](https://github.com/AdamEzzat1/CJC/blob/master/CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0043%20cjc-causal%20v0.1%20%E2%80%94%20Propensity%20Score%2C%20IV%2C%20Double%20ML.md)
- Phase 2 handoff §2 — release-engineering process (workspace bump, CI
  matrix, publication order)
- [`cjc-locke`](https://crates.io/crates/cjc-locke) — upstream data-skepticism
  layer whose `LockeReport` every estimator consumes

## License

MIT
