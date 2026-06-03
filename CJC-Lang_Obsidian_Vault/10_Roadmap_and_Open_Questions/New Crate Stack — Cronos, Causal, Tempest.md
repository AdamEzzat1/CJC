---
title: "New Crate Stack — Cronos, Causal, Tempest"
tags: [roadmap, planning, crates, causal-inference, time-series, probabilistic-programming, handoff]
status: 📋 Planned — three sibling crates queued for separate sessions
date: 2026-06-02
date-modified: 2026-06-02
---

# New Crate Stack — Cronos, Causal, Tempest

Handoff for three new sibling crates queued to follow the [[Locke v0.1.10 — An Epistemic Layer for Data Validation|Locke v0.8 release]] (cjc-locke 0.1.11 on crates.io, cjc-locke 0.1.2 on PyPI). Each crate ships in its own session; this document is the per-session brief and the cross-crate coordination doc.

## 0. TL;DR

Three new sibling crates extend CJC-Lang's library surface into inference and decision layers that the current stack (cjc-data → cjc-ad → cjc-runtime → cjc-locke → cjc-abng) does not yet cover:

| Crate         | Domain                                  | Primary value claim                              | Sessions |
| ------------- | --------------------------------------- | ------------------------------------------------ | -------- |
| `cjc-causal`  | Formal causal inference                 | Byte-identically-reproducible treatment-effect estimates | ~3       |
| `cjc-cronos`  | Time-series analysis + forecasting      | Byte-identical backtests across runs / library upgrades  | ~3       |
| `cjc-tempest` | Probabilistic programming + MCMC        | Byte-identical posterior chains (HMC / NUTS)             | ~4-6     |

Names verified available on crates.io (2026-06-02). `cjc-causal`, `cjc-cronos`, `cjc-tempest` all unclaimed. The bare `kronos` is taken by an unrelated time-expression library — sticking with `cronos` (and the `cjc-` prefix anyway) avoids brand confusion.

**Build order recommendation**: `cjc-causal` → `cjc-cronos` → `cjc-tempest`. Lowest-scope-highest-fit first; biggest-scope-highest-payoff last. Each ships independently; downstream crates are not blocked by upstream crates.

Each crate's handoff below is organized by the six stack roles from [[CLAUDE]]:

1. **Lead Language Architect** — language semantics, type system soundness, feature design.
2. **Compiler Pipeline Engineer** — Lexer → Parser → AST → HIR → MIR → Exec.
3. **Runtime Systems Engineer** — memory model, GC/NoGC boundary, dispatch, builtins.
4. **Numerical Computing Engineer** — deterministic BLAS, SIMD, accumulator correctness, AD.
5. **Determinism & Reproducibility Auditor** — bit-identical output across runs and platforms.
6. **QA Automation Engineer** — test infrastructure, parity gates, regression prevention.

Test discipline applies uniformly to all three: **unit + integration + proptest + bolero fuzz** are non-negotiable. See §6 for the cross-crate test contract.

---

## 1. Why these three, why now, what they share

The [[Locke v0.1.10 — An Epistemic Layer for Data Validation|Locke v0.8 release post]] ended with an explicit framing: *Locke is "where to look", not "what to do".* The analyst still applies domain triage, formal estimation, and decision logic on top of Locke's audited DataFrame. The three crates in this handoff are the natural decision-layer triad:

- **`cjc-causal`** answers *"what would happen if I intervened?"* — formal estimation of treatment effects under declared identifying assumptions.
- **`cjc-cronos`** answers *"what will happen next?"* — point and interval forecasts from time-indexed data.
- **`cjc-tempest`** answers *"how uncertain am I, and why?"* — posterior inference over parameters via byte-identically-reproducible MCMC.

All three are *uniquely CJC-shaped* because each one's headline value claim depends on determinism:

- Causal estimates that don't reproduce across runs hide identification fragility.
- Forecasts that don't reproduce across library upgrades break production backtests.
- Posterior chains that don't reproduce across runs make Bayesian model selection unreliable.

The Python ecosystem incumbents (DoWhy / EconML / CausalML for causal, statsmodels / Prophet / darts for time series, PyMC / Stan / Turing.jl for PPL) do not provide structural determinism. CJC-Lang's SplitMix64 + Kahan + BTreeMap discipline can — and **that is the publishable claim per crate**.

What each crate shares with the others:

- **Workspace member** under `crates/cjc-{causal,cronos,tempest}/` with the standard `Cargo.toml` (`version = "0.1.0"`, `edition = "2021"`, `license = "MIT"`).
- **Path-deps on** cjc-data, cjc-ad, cjc-runtime, cjc-repro at minimum. cjc-locke as optional for findings emission. cjc-abng as optional for belief composition.
- **No external runtime dependencies** outside the cjc-* workspace. flate2 is acceptable if compression is needed (already in workspace tree for cjc-vizor).
- **Public API surface that emits Locke-compatible findings** when assumptions are violated (positivity / overlap / weak instruments / drift between train and forecast / posterior-data conflict). The findings reuse `cjc_locke::report::ValidationFinding` so reports compose.
- **Three-file test layout** (see §6): unit tests in the crate's `lib.rs`, integration tests in `tests/<crate_name>/`, fuzz targets cohabiting with integration tests via the `bolero` pattern already used in `tests/locke/locke_fuzz.rs`.
- **Python bindings deferred** until the Rust v0.1 ships. Each crate gets a `python/cjc_{name}/` mirror via maturin in its own follow-up session (matching the cjc-locke-py pattern).

---

## 2. `cjc-causal` — formal causal inference

**Status**: 📋 Planned — first to ship per recommended build order.
**Scope**: ~3 sessions for v0.1 (propensity score matching + IV regression + double ML).
**Dependencies**: cjc-data, cjc-ad, cjc-runtime, cjc-repro, cjc-locke.
**Architectural fit**: closes the gap [[ADR-0028 Locke Data Skepticism Layer]] explicitly leaves open ("no warning kind is named 'Cause' or 'Effect'"). Locke validates → cjc-causal estimates → ABNG composes the belief.

### 2.1 Lead Language Architect — responsibilities

- **Settle the public API shape** before code lands. Pick one of:
  - **Method-specific structs** (`PropensityScoreMatcher`, `IVRegression`, `DoubleMLEstimator`, each with their own `fit()` / `estimate()` methods). Cleaner type story, more code duplication.
  - **Unified `CausalEstimator` trait** with `fit(df, treatment, outcome, covariates) -> Result<EffectEstimate, CausalError>`. Less code; trait method dispatch makes some method-specific knobs awkward.
  - **Recommendation**: method-specific structs for v0.1. Reach for the unified trait in v0.2 if a real cross-method abstraction emerges.
- **Define `IdentificationAssumption`** as an explicit enum that callers MUST construct and pass to estimators. No default assumptions — every estimator's `estimate()` takes a `Vec<IdentificationAssumption>` argument so the analyst's declared assumptions become part of the report. Variants: `Unconfoundedness`, `Positivity`, `ExcludabilityOfInstrument`, `MonotonicityOfInstrument`, `ParallelTrends`, `LocalRandomization`, `NoInterference (SUTVA)`.
- **Output type contract**: `EffectEstimate { point: f64, std_error: f64, ci_lower: f64, ci_upper: f64, n_treated: u64, n_control: u64, assumptions_declared: Vec<IdentificationAssumption>, balance_diagnostics: Option<BalanceReport>, identifier: FingerprintId }`. The `identifier` is content-addressed over (estimator type, treatment, outcome, covariates, assumptions) — two runs that produced the same estimate carry the same `identifier`.
- **Composition with cjc-locke**: decide whether causal estimators *call* Locke internally (validating their input automatically) or *consume* a Locke report passed in by the caller. Recommendation: caller passes the Locke report explicitly. Estimators check for known-fatal findings (E9001 ≥ 0.30 on treatment / outcome, E9009 missing on continuous covariates, E9060 leakage) and refuse to run with a `CausalError::DataQualityRefusal { findings: Vec<ValidationFinding> }`.
- **No language-level builtins in v0.1.** Library-only Rust API. Cjcl language integration is a v0.2 question, not a v0.1 blocker.

### 2.2 Compiler Pipeline Engineer — responsibilities

- **Minimal involvement in v0.1.** No lexer / parser / AST / HIR / MIR changes required because cjc-causal does not introduce new language-level syntax. The crate is invoked through `cargo run` or by linking the library — not through `cjcl` source files.
- **v0.2+ open question**: if cjc-causal becomes accessible from `.cjcl` source via builtins (see §2.3), the compiler pipeline gets one new node type per estimator. Not in v0.1 scope. Document as a follow-up in [[Open Questions]].

### 2.3 Runtime Systems Engineer — responsibilities

- **Builtin registration is deferred to v0.2.** v0.1 stays Rust-library-only. When the time comes:
  - One builtin per estimator entry point (`propensity_score_match`, `iv_regress`, `double_ml`).
  - Each takes a DataFrame value, treatment column name, outcome column name, covariate column names, and the assumption set as a list-of-strings or a structured enum.
  - Wired in three places per the [[CLAUDE|wiring pattern]]: `cjc-runtime/src/builtins.rs`, `cjc-eval/src/lib.rs`, `cjc-mir-exec/src/lib.rs`.
- **Reuse existing primitives**: `cjc_runtime::hypothesis::logistic_regression` for propensity scoring; `cjc_runtime::ml::auc_roc` and `cjc_runtime::ml::train_test_split` for cross-fitting in double ML. No new linalg primitives required for v0.1.
- **No new tensor variants needed.** All causal estimators work on existing `Column::Float` / `Column::Bool` / `Column::Int` data via cjc-data's DataFrame.

### 2.4 Numerical Computing Engineer — owns this crate

- **Propensity score matching** (session 1 deliverable):
  - IRLS-based logistic regression (reuse `cjc_runtime::hypothesis::logistic_regression`) for propensity score.
  - Nearest-neighbor matching with caliper. Distance is `|logit(p_treated) - logit(p_control)|`.
  - Tie-breaking: when two control units have identical logits (or within `f64::EPSILON`), break ties by ascending row index. NEVER by hash order. Document this in the function's docstring as the determinism contract.
  - Caliper width: default 0.2 standard deviations of the propensity logit (the canonical Austin 2011 recommendation). Configurable.
  - Output: matched pairs as `Vec<(usize, usize)>` plus an `EffectEstimate`.
- **Instrumental variables (2SLS)** (session 2 deliverable):
  - Two-stage least squares via direct linear algebra (no IRLS needed). Stage 1: regress treatment on instrument + covariates. Stage 2: regress outcome on fitted treatment + covariates.
  - Weak-instrument F-statistic computed and surfaced as evidence on the `EffectEstimate`. F < 10 (Stock-Yogo critical value) triggers a Locke-compatible finding `E9100` (claim a new code).
  - Heteroskedasticity-robust standard errors (Huber-White / HC1 sandwich estimator).
- **Double machine learning (DML)** (session 3 deliverable):
  - Orthogonal moment functions for partially linear regression and interactive regression models (Chernozhukov et al. 2018).
  - Cross-fitting via K-fold sample splitting (`cjc_runtime::ml::kfold_indices`). K=5 default.
  - Nuisance functions fit via `cjc_ad::GradGraph`-trained MLPs (per the [[Phase 3c — Language-Level GradGraph Primitives]] pattern). Treatment-model and outcome-model are separate networks.
  - Output: orthogonalised effect estimate + standard error from the variance of the orthogonal moments.
- **All accumulations Kahan-summed.** Reuse `cjc_runtime::accumulator::KahanAccumulatorF64`. No raw `sum()` on `Vec<f64>` in any reduction.
- **All map iteration BTreeMap.** When tabulating matched pairs by stratum, when building covariate-balance breakdowns, use BTreeMap. No HashMap anywhere.

### 2.5 Determinism & Reproducibility Auditor — owns the determinism story

- **Bootstrap confidence intervals**: the most likely source of cross-run drift. The bootstrap RNG must be a `cjc_repro::Rng` (SplitMix64) seeded from the caller. Document that the seed is part of the `EffectEstimate.identifier` content hash.
- **Sample-splitting in DML**: the K-fold split is RNG-driven (Fisher-Yates inside `cjc_runtime::ml::kfold_indices`). Confirm the seed is threaded explicitly, not derived from system entropy.
- **Tie-breaking in nearest-neighbor matching**: every tie MUST resolve to ascending row index. Audit the matching loop for any `.iter()` over a HashMap or any reliance on iteration order from an unsorted collection.
- **`logit()` and `expit()` numerical guards**: clamp to `[1e-10, 1 - 1e-10]` before taking logit (the same guard already used in `cjc-runtime/src/hypothesis.rs::logistic_regression`). This prevents `inf` propagation that would silently corrupt downstream estimates.
- **Cross-run byte-identity proptest**: same input frame + same seed + same assumption set ⇒ byte-identical `EffectEstimate` (point, std_error, identifier). Required as part of the test surface (§2.6).
- **Cross-platform parity**: the Linux/macOS Test Suite must produce the same `EffectEstimate.identifier` as Windows for an identical input. This is the [[ADR-0002 Kahan Accumulator|Kahan guarantee]] in practice. If platform parity fails, the build does NOT ship.

### 2.6 QA Automation Engineer — responsibilities

| Test type           | Location                                             | Required coverage                                                 |
| ------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| **Unit**            | `crates/cjc-causal/src/lib.rs` (in-module `tests`)   | Each estimator: small synthetic dataset, closed-form known answer, assertion on point estimate within `f64::EPSILON * 100`. Includes balance check helpers, caliper edge cases (empty matched set), IV F-stat, DML orthogonal-moment closure. |
| **Integration**     | `tests/causal/` (registered in root `Cargo.toml` `[[test]]`) | End-to-end on hand-crafted DataFrames where the true effect is known by construction. Covers all three estimators. Includes one "data quality refusal" test where the input has E9001 ≥ 0.50 and the estimator MUST return `CausalError::DataQualityRefusal`. |
| **Proptest**        | `tests/causal/causal_proptest.rs`                    | (1) Same input + same seed ⇒ byte-identical `EffectEstimate`. (2) `EffectEstimate.identifier` is stable across two runs. (3) Adding a constant to the outcome shifts the estimate by exactly that constant. (4) Adding a constant to a covariate does not change the estimate. (5) Doubling the sample size never decreases the absolute number of matched pairs (monotonicity). |
| **Bolero fuzz**     | `tests/causal/causal_fuzz.rs`                        | (1) Arbitrary numeric covariates: estimator returns `Ok` or structured `Err`, never panics. (2) Arbitrary treatment vectors (including all-zero, all-one, all-NaN): no panic. (3) Arbitrary caliper widths (negative, zero, infinite, NaN): caught at config time, no panic. |
| **Parity (Locke)**  | `tests/causal/causal_locke_parity.rs`                | Causal estimator output is byte-identical between `lib::estimate(df, ...)` direct call and a `validate_then_estimate(df, ...)` Locke-piped variant. |
| **Determinism**     | `tests/causal/causal_determinism.rs`                 | Two runs over the same synthetic frame produce byte-identical `EffectEstimate` JSON (use a future `to_json()` method matching cjc-locke's pattern). Cross-platform CI verifies this on Linux + macOS + Windows. |

**Required counts before merge**: unit ≥ 25, integration ≥ 12, proptest properties ≥ 5 (each with `cases = 256` default), bolero targets ≥ 3.

### 2.7 What v0.1 does NOT do (scope discipline)

- Full do-calculus / Pearl identification — deferred to v0.2.
- Structural equation modelling — orthogonal scope; no SEM story in CJC-Lang yet.
- Mediation analysis — pick a side later; sits between propensity scoring and SEM.
- Regression discontinuity — defer to v0.2 (the bandwidth-selection literature is its own rabbit hole).
- Difference-in-differences — defer to v0.2 (modern DiD needs the staggered-adoption fixes from Callaway-Sant'Anna; significant scope).
- Synthetic control method — deferred indefinitely.
- Causal forests / generalized random forests — deferred indefinitely.

The scope-discipline summary: v0.1 ships *the three estimators econometrics teaching tracks most often cover first*, plus the data-quality-refusal hook into cjc-locke. v0.2 expands to RD + DiD. v0.3 introduces do-calculus.

---

## 3. `cjc-cronos` — deterministic time-series and forecasting

**Status**: 📋 Planned — second in build order.
**Scope**: ~3 sessions for v0.1 (ETS + ARIMA + Kalman filter + backtest framework).
**Dependencies**: cjc-data, cjc-ad, cjc-runtime, cjc-repro, cjc-locke.
**Architectural fit**: cjc-locke has `drift::compare(train, test)` for distributional drift; cjc-cronos is the natural producer of `test`. Composes via "fit on train → forecast → cjc-locke drift-compares the forecast distribution to test".

### 3.1 Lead Language Architect — responsibilities

- **Settle the time-series representation.** Two options:
  - **`TimeSeries<T>` wrapper struct** around a DataFrame with a designated time column and a designated value column.
  - **Plain DataFrame** with a column-name argument to every function call.
  - **Recommendation**: `TimeSeries<f64>` wrapper for the v0.1 API. Carries the time index + value array + a `Frequency` enum (`Hourly`, `Daily`, `Weekly`, `Monthly`, `Quarterly`, `Annual`, `Irregular`). The wrapper's `Drop` is a no-op (no GC entanglement); construction from a DataFrame is a fallible `TimeSeries::from_dataframe(df, time_col, value_col)`.
- **Model output contract**: `Forecast { horizon: usize, point_estimates: Vec<f64>, lower_bound: Vec<f64>, upper_bound: Vec<f64>, confidence_level: f64, fitted_model_id: FingerprintId }`. The `fitted_model_id` is content-addressed over (model class, hyperparameters, training data fingerprint, seed) — two backtests over the same data with the same model emit the same `fitted_model_id`.
- **Decompose vs forecast**: STL decomposition (`Decomposition { trend, seasonal, residual }`) is its own output type, separate from forecasting.
- **Backtest framework API**: `backtest(model, ts, initial_window, step) -> BacktestReport`. The `BacktestReport` is the headline output — it's content-addressed, contains per-horizon MAE / MAPE / RMSE, and is the artifact the determinism claim is anchored to.

### 3.2 Compiler Pipeline Engineer — responsibilities

- **Minimal involvement in v0.1.** No language-level changes.
- **v0.2 open question**: should `.cjcl` source support a `time_series` literal type? Probably not — the wrapper struct stays in Rust. Document as a follow-up.

### 3.3 Runtime Systems Engineer — responsibilities

- **Builtin registration for v0.2.** Functions to expose: `arima_fit`, `ets_fit`, `kalman_smooth`, `stl_decompose`, `forecast`, `backtest`. Each follows the [[CLAUDE|three-place wiring pattern]].
- **Reuse `cjc_runtime::stats` and `cjc_runtime::ml`**: ACF / PACF helpers, train-test split for backtest, AUC for classification-flavored time-series tasks if relevant.
- **No new tensor variants.** `cjc_data::Column::Float` with `cjc_data::Column::DateTime` as the time index handles the v0.1 surface.

### 3.4 Numerical Computing Engineer — owns this crate

- **Exponential smoothing (ETS)** — session 1:
  - Simple, double (Holt), triple (Holt-Winters) variants.
  - Additive and multiplicative seasonality.
  - Initialization: average of first season for level, slope from first two periods for trend, ratios for seasonal indices.
  - Smoothing parameter optimisation via grid search OR via `cjc_ad::GradGraph` on the negative log-likelihood. v0.1 ships grid search (faster, deterministic by construction); gradient-based is v0.2.
  - State updates: classical recursive form `l_t = α y_t + (1-α)(l_{t-1} + b_{t-1})`. Every step Kahan-summed.
- **ARIMA / SARIMA** — session 2:
  - AR coefficients via Yule-Walker estimation initially; refine via conditional sum of squares (CSS) maximum likelihood.
  - MA coefficients via Hannan-Rissanen two-step procedure.
  - Differencing for integration order `d`.
  - Forecast intervals from the model's MA(∞) representation.
  - Note: `auto_arima` (parameter search) is its own beast. v0.1 ships hand-specified `(p, d, q)`; auto-search is v0.2.
- **State-space / Kalman filter and smoother** — session 3:
  - Forward filter (Joseph form for numerical stability).
  - Backward (RTS) smoother for retrospective state estimates.
  - Local-level + local-linear-trend + Basic Structural Model (BSM) variants.
  - All matrix operations through `cjc_runtime::linalg`; no FMA.
- **STL decomposition** — session 1 or 2 depending on slot:
  - Cleveland 1990 STL algorithm.
  - Per-iteration loess smoother (Kahan-summed weighted sum).
  - Cycle subseries for seasonal extraction.

### 3.5 Determinism & Reproducibility Auditor — owns the determinism story

- **Forecast interval randomization MUST be off by default.** Bootstrap interval generation (`forecast_with_bootstrap`) is gated behind an explicit seed parameter, never the system clock.
- **Kalman recursions are the highest determinism risk.** Every multiplication and addition propagates error. Joseph form is mandated specifically because it's symmetric-positive-definite-preserving — the standard `(I - K H) P` form drifts to non-PSD under accumulated rounding.
- **STL convergence test**: STL iterates until the trend changes by < ε between iterations. Audit that ε is the same constant across runs (`f64::EPSILON * 1e4` is a reasonable default) and that the convergence test is `(new - old).abs() < eps`, not `(new - old) < eps` (the absolute value matters).
- **No FMA in ARIMA likelihood evaluation.** The compiled binary's `RUSTFLAGS` must omit `target-feature=+fma`. The crate's CI runs a test that verifies `std::arch::is_x86_feature_detected!("fma")` is not relied on.
- **Cross-platform parity**: same backtest on the same input produces identical `BacktestReport.fitted_model_id` across Linux / macOS / Windows.

### 3.6 QA Automation Engineer — responsibilities

| Test type           | Location                                             | Required coverage                                                 |
| ------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| **Unit**            | `crates/cjc-cronos/src/lib.rs` (in-module `tests`)   | ETS coefficient updates on hand-traceable 5-row series. ARIMA Yule-Walker against textbook AR(2) example. Kalman filter on a known linear-Gaussian state-space. STL decomposition on a sinusoid + trend (decomposition should recover both). Forecast confidence intervals at known α levels. |
| **Integration**     | `tests/cronos/`                                      | Full backtest on a synthetic series with known generative process; verify `BacktestReport` MAE within tolerance. Compose with cjc-locke: fit, forecast, drift-compare against held-out test set. |
| **Proptest**        | `tests/cronos/cronos_proptest.rs`                    | (1) Forecast of constant series is constant. (2) Forecast of linear-trend series is linear (extrapolated). (3) ARIMA(0,0,0) fit returns the mean. (4) STL `trend + seasonal + residual == original` within `f64::EPSILON * n_rows`. (5) Two backtests on the same series with the same seed return byte-identical `BacktestReport.fitted_model_id`. |
| **Bolero fuzz**     | `tests/cronos/cronos_fuzz.rs`                        | (1) Arbitrary `Vec<f64>` series (including all-NaN, all-zero, very small, very large): fit + forecast either succeed or return structured error, never panic. (2) Arbitrary forecast horizons (zero, huge, negative): caught at API boundary. (3) Arbitrary ARIMA `(p, d, q)` triples: invalid combinations caught at config time. |
| **Parity (Locke)**  | `tests/cronos/cronos_locke_parity.rs`                | Forecast distribution piped into `cjc_locke::drift::compare` produces stable findings across runs. |
| **Determinism**     | `tests/cronos/cronos_determinism.rs`                 | Two backtest runs on the same series produce byte-identical `BacktestReport` JSON. Cross-platform CI verifies this. |

**Required counts before merge**: unit ≥ 30, integration ≥ 15, proptest properties ≥ 5, bolero targets ≥ 3.

### 3.7 What v0.1 does NOT do

- **`auto_arima`** — defer to v0.2.
- **Gradient-based ETS parameter fitting** — defer to v0.2 (v0.1 uses grid search for determinism by construction).
- **Multivariate VAR / VECM** — defer to v0.2.
- **Prophet-style trend-changepoint detection** — out of scope; Prophet's model is opinionated and bundling it would dilute cjc-cronos's identity.
- **Neural forecasting (DeepAR / N-BEATS / TFT)** — orthogonal scope; might land in a `cjc-neural-forecast` sibling crate later.

---

## 4. `cjc-tempest` — probabilistic programming with deterministic MCMC

**Status**: 📋 Planned — third in build order (highest scope, highest payoff).
**Scope**: ~4-6 sessions for v0.1 (Metropolis sampler → HMC → NUTS as three release milestones; or a single v0.1 with all three if appetite permits).
**Dependencies**: cjc-ad (CRITICAL — reverse-mode AD required for HMC), cjc-runtime, cjc-repro, cjc-locke (data validation pre-inference), cjc-abng (graph-structured model representation).
**Architectural fit**: this is the "killer feature" of the three crates. No existing PPL provides byte-identical posterior chains across runs. The unique selling proposition is structural: Bayesian model selection, posterior predictive checks, sensitivity analysis all become reproducible.

### 4.1 Lead Language Architect — responsibilities

- **Settle the model representation.** Three options:
  - **Closure-based**: user supplies `log_posterior: Fn(&[f64]) -> f64`. Simple; opaque to inspection.
  - **DAG-based via cjc-abng**: user constructs a `BayesianNetwork` and tempest samples it. Composes with existing tooling; more setup overhead.
  - **DSL block in `.cjcl`**: `model { theta ~ Normal(0, 1); y ~ Bernoulli(sigmoid(theta * x)); }`. Cleanest user experience; biggest implementation effort.
  - **Recommendation**: closure-based for v0.1, DAG-based as a thin wrapper. DSL block is a v0.3+ language-level proposal (separate ADR).
- **Sampler trait**: `pub trait Sampler { fn step(&mut self, state: &mut State, rng: &mut Rng) -> AcceptResult; }`. Concrete impls: `MetropolisHastings`, `HamiltonianMonteCarlo`, `NoUTurnSampler`. The trait is intentionally low-level; higher-level `sample(n_chains, n_iter, ...)` lives outside the trait.
- **Posterior output**: `PosteriorSamples { chains: Vec<Vec<Vec<f64>>>, n_chains: usize, n_samples_per_chain: usize, n_dim: usize, diagnostics: ConvergenceDiagnostics, content_hash: FingerprintId }`. The `content_hash` is the *whole point* — published Bayesian analyses cite this hash, anyone re-running the analysis gets the same hash.
- **Diagnostics output**: `ConvergenceDiagnostics { r_hat: Vec<f64>, ess_bulk: Vec<f64>, ess_tail: Vec<f64>, divergences: u64, n_max_treedepth: u64 }`. R-hat and ESS computation must be deterministic (Vehtari et al. 2021 split-rank-normalised formulation; no HashMap).

### 4.2 Compiler Pipeline Engineer — responsibilities

- **v0.1: zero involvement** beyond what cjc-ad already requires.
- **v0.3+ open question**: the `model { ... }` DSL block. New AST node, new HIR lowering, new MIR codegen. This is the largest single piece of compiler work the proposed stack implies — it deserves its own ADR. Document the need as part of v0.1's "Open Questions" section.

### 4.3 Runtime Systems Engineer — responsibilities

- **Reuse `cjc-ad::GradGraph`** for gradient computation. HMC requires reverse-mode AD over the log posterior — `cjc-ad` already provides this via the arena-based GradGraph (per [[ADR-0016 Language-Level GradGraph Primitives]]).
- **Posterior storage via cjc-snap.** Posterior chains can be large (10000 samples × 50 parameters × 4 chains = 16MB per dataset). Serialize to a content-addressed `.cjcsnap` file for sharing.
- **No new tensor variants needed.** Posterior samples are `Vec<Vec<Vec<f64>>>` of shape `(chains, samples, parameters)`.

### 4.4 Numerical Computing Engineer — owns this crate

- **Metropolis-Hastings** — session 1 (warm-up before HMC):
  - Symmetric proposal kernel: multivariate normal with adapted covariance.
  - Proposal covariance adapts during warm-up via Welford online covariance estimation; freezes for sampling phase.
  - Acceptance ratio: `exp(log_p_new - log_p_current)`, clamped to `[0, 1]`.
  - This is the determinism-warmup deliverable: get the seed-flow audit done on the simpler sampler before tackling HMC.
- **Hamiltonian Monte Carlo** — session 2-3:
  - Leapfrog integrator with caller-specified step size `epsilon` and trajectory length `L`.
  - Kinetic energy uses an identity mass matrix in v0.1 (mass adaptation is v0.2).
  - Energy error tracking: divergences declared when `|H_new - H_initial| > 1000`.
  - Reverse-mode AD via cjc-ad: log-posterior closure is called once per leapfrog step; gradient computation via `GradGraph::backward()`.
- **No-U-Turn Sampler (NUTS)** — session 3-4:
  - Algorithm 6 from Hoffman & Gelman 2014 (efficient NUTS with dual averaging).
  - Dual averaging for step-size adaptation during warm-up.
  - Recursive tree building with no-U-turn termination criterion.
  - This is the headline. Once it works deterministically, the unique-selling-proposition claim is real.
- **Convergence diagnostics** — session 4 if NUTS spills over, otherwise session 3:
  - Split-rank-normalised R-hat (Vehtari et al. 2021).
  - Effective sample size: bulk-ESS and tail-ESS via autocorrelation summation through Geyer's initial monotone sequence estimator.
- **All accumulation Kahan-summed.** No exceptions. The leapfrog integrator is particularly sensitive — even small per-step rounding error compounds over a trajectory.

### 4.5 Determinism & Reproducibility Auditor — owns the determinism story (this is the whole point of the crate)

- **Every RNG draw routes through `cjc_repro::Rng` (SplitMix64).** Auditor maintains a list of every site where randomness is consumed:
  1. Initial state per chain.
  2. Momentum draw per leapfrog kick (HMC, NUTS).
  3. Acceptance/rejection coin flip per proposal.
  4. Direction choice per NUTS tree expansion (left vs right).
  5. Slice sample for NUTS (Hoffman & Gelman 2014 §3.1).
  6. Adaptation-phase RNG (warmup); SEPARATE seed from sampling-phase RNG.
- **Reproducibility lock**: same seed + same model + same input data ⇒ byte-identical `PosteriorSamples.content_hash`. This is the headline proptest (§4.6).
- **Cross-platform parity**: same posterior content hash on Linux / macOS / Windows. The math is the hardest of any of the three crates to keep cross-platform stable; expect to debug.
- **No HashMap in any internal data structure**. Adaptation history (step size, mass diagonal) goes in BTreeMap or Vec with explicit ordering.
- **Document the seed-flow diagram** as part of the crate's `src/lib.rs` module documentation. A future contributor MUST be able to trace where every random number comes from. This is the cjc-tempest equivalent of cjc-locke's [[ADR-0028 Locke Data Skepticism Layer|determinism note]].

### 4.6 QA Automation Engineer — responsibilities

| Test type           | Location                                             | Required coverage                                                 |
| ------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| **Unit**            | `crates/cjc-tempest/src/lib.rs` (in-module `tests`)  | Metropolis on a 1-D standard Gaussian (analytic mean = 0, variance = 1; sampler converges within tolerance). HMC on a 2-D banana distribution (Rosenbrock-shaped; samples cover known density). NUTS on an 8-Schools hierarchical model (the canonical PPL test). R-hat for two converged chains is < 1.01. ESS computation against closed-form for AR(1) series. |
| **Integration**     | `tests/tempest/`                                     | End-to-end Bayesian logistic regression: synthetic data with known coefficient, posterior mean recovers the coefficient within posterior SD. Compose with cjc-locke (validate input before inference); compose with cjc-abng (graph-structured model evaluation). |
| **Proptest**        | `tests/tempest/tempest_proptest.rs`                  | (1) Same seed + same model + same data ⇒ byte-identical `content_hash`. (2) Samples are always in the model's support (e.g., positive constraints stay positive). (3) ESS ≤ n_samples_per_chain (definitional). (4) R-hat for a single chain equals 1.0 by definition. (5) Doubling n_chains does not change per-chain content hashes. |
| **Bolero fuzz**     | `tests/tempest/tempest_fuzz.rs`                      | (1) Arbitrary log-posterior closures (including ones that return NaN, Inf): sampler returns structured error, never panics. (2) Arbitrary initial states (including NaN-laden, very large): sampler refuses or recovers. (3) Arbitrary step-size / trajectory-length combinations: invalid combos caught at config time. |
| **Parity (cjc-ad)** | `tests/tempest/tempest_ad_parity.rs`                 | HMC's gradient via cjc-ad GradGraph matches numerical-differentiation gradient on a synthetic posterior, within `1e-6` for double precision. |
| **Determinism**     | `tests/tempest/tempest_determinism.rs`               | Two NUTS runs over the same model + same seed produce byte-identical `content_hash`. Cross-platform CI verifies this on Linux + macOS + Windows. This is the **headline test**. If it fails, do not ship. |
| **Reference**       | `tests/tempest/tempest_stan_reference.rs` (optional) | Compare posterior means against Stan's output for the 8-Schools model. Tolerance: 2 posterior SDs. Not for byte-identity (different samplers, different RNGs) — for *direction-of-effect* validation. |

**Required counts before merge**: unit ≥ 25, integration ≥ 10, proptest properties ≥ 5, bolero targets ≥ 3. The determinism test is non-negotiable — if it fails CI on any platform, the release does not ship.

### 4.7 What v0.1 does NOT do

- **Variational inference (ADVI / SVI)** — defer to v0.2.
- **Posterior predictive sampling** — small extension; either ship in v0.1.1 or wait for v0.2.
- **Reversible-jump MCMC** — orthogonal; defer indefinitely.
- **Sequential Monte Carlo** — defer indefinitely.
- **`model { ... }` DSL block in `.cjcl`** — v0.3+ (separate ADR).
- **Marginal likelihood (Bayes factor) computation** — bridge sampling is its own discipline; defer to v0.2 or later.

---

## 5. Cross-crate shared infrastructure

### 5.1 Workspace registration

Each crate registers as a workspace member in the root `Cargo.toml`:

```toml
[workspace]
members = [
    # ... existing entries ...
    "crates/cjc-causal",
    "crates/cjc-cronos",
    "crates/cjc-tempest",
]

[workspace.dependencies]
# ... existing entries ...
cjc-causal  = { path = "crates/cjc-causal",  version = "0.1.0" }
cjc-cronos  = { path = "crates/cjc-cronos",  version = "0.1.0" }
cjc-tempest = { path = "crates/cjc-tempest", version = "0.1.0" }
```

Root `[dependencies]` reference each crate only if root-level integration tests use them. Each new crate's release version aligns with the workspace version (currently `0.1.11`; new crates start at `0.1.0` and bump in lockstep on subsequent workspace releases).

### 5.2 Determinism contract (shared)

All three crates obey the [[CLAUDE|Prime Directives]] determinism contract:

- Floating-point reductions: Kahan or BinnedAccumulator. No raw `sum()`.
- Map iteration: BTreeMap or BTreeSet. No HashMap, no HashSet with default hasher.
- RNG: `cjc_repro::Rng` (SplitMix64) with explicit seed threading. Never `rand::thread_rng()`.
- SIMD: no FMA (fused multiply-add). Build flags omit `target-feature=+fma`.
- Parallel execution: results identical regardless of thread count.
- Cross-platform: identical output bytes on Linux + macOS + Windows for the same input.

Violations of this contract fail CI. The Determinism Auditor role is the gatekeeper for any PR touching these crates.

### 5.3 Findings emission convention

When any of these crates surfaces a data-quality / model-quality concern, it emits a `cjc_locke::report::ValidationFinding` with a new E-code in the **E9100..=E9499** range (between cjc-locke's E9001..=E9112 and the custom-detector E9500..=E9999 range from [[ADR-0041 Locke v0.8 — Custom Detector Extension Layer]]). Suggested code assignments:

- **E9100..=E9199 — cjc-causal**:
  - E9100 — weak instrument detected (F-statistic < 10)
  - E9101 — positivity violation (propensity score near 0 or 1)
  - E9102 — covariate imbalance after matching (standardised mean difference > 0.10)
  - E9103 — overlap failure (untreated has no matched treated within caliper)
  - E9104 — orthogonality violation in double ML
- **E9200..=E9299 — cjc-cronos**:
  - E9200 — non-stationarity detected (ADF p > 0.05) before ARIMA fit
  - E9201 — seasonal pattern detected on a non-seasonal model
  - E9202 — forecast confidence interval exceeds historical range (extrapolation warning)
  - E9203 — Kalman filter innovation variance non-PSD (numerical failure)
- **E9300..=E9399 — cjc-tempest**:
  - E9300 — divergent transitions (HMC / NUTS energy error)
  - E9301 — R-hat > 1.01 on any parameter (convergence failure)
  - E9302 — ESS bulk < 400 (insufficient effective sample size)
  - E9303 — max tree depth hit (NUTS tree expansion stopped before U-turn)

The Locke Custom Detector Extension Layer can wrap these as belief-axis-affecting findings if the user wants; otherwise they appear as plain findings in the report.

### 5.4 Python bridge (deferred but planned)

Each crate gets a `python/cjc_{name}/` mirror in a follow-up session per crate, modelled after the `cjc-locke-py` pattern from the [[Locke v0.1.10 — An Epistemic Layer for Data Validation|v0.8 release]]. Out of v0.1 scope for the Rust release; queued for v0.1.1 or v0.2 of each crate.

---

## 6. Cross-crate test discipline

Every PR to any of these crates must pass:

| Discipline                  | Mechanism                                                   | Cadence                  |
| --------------------------- | ----------------------------------------------------------- | ------------------------ |
| **Unit**                    | `cargo test -p cjc-{causal,cronos,tempest} --lib --release` | Every commit             |
| **Integration**             | `cargo test --test {causal,cronos,tempest} --release`       | Every commit             |
| **Proptest**                | `cargo test --test {causal,cronos,tempest}_proptest --release` | Every commit (256 cases each) |
| **Bolero fuzz**             | `cargo test --test {causal,cronos,tempest}_fuzz --release`  | Every commit (default budget) |
| **Determinism (cross-run)** | `tests/{causal,cronos,tempest}_determinism.rs`              | Every commit; HEADLINE   |
| **Cross-platform parity**   | CI matrix: ubuntu-latest, macos-latest, windows-latest      | Every push to main       |
| **Reference parity (Locke)**| `tests/{name}_locke_parity.rs`                              | Every PR touching either |

Tests are registered in the root `Cargo.toml` `[[test]]` table per [[Locke v0.1.10 — An Epistemic Layer for Data Validation|the cjc-locke pattern]]. Example:

```toml
[[test]]
name = "causal"
path = "tests/causal/mod.rs"

[[test]]
name = "cronos"
path = "tests/cronos/mod.rs"

[[test]]
name = "tempest"
path = "tests/tempest/mod.rs"
```

The proptest and bolero fuzz files are sub-modules of these (matching the `tests/locke/locke_proptest.rs` + `tests/locke/locke_fuzz.rs` layout).

### 6.1 Required minimum test counts before merge

| Crate         | Unit | Integration | Proptest properties | Bolero targets |
| ------------- | ---- | ----------- | ------------------- | -------------- |
| `cjc-causal`  | ≥ 25 | ≥ 12        | ≥ 5                 | ≥ 3            |
| `cjc-cronos`  | ≥ 30 | ≥ 15        | ≥ 5                 | ≥ 3            |
| `cjc-tempest` | ≥ 25 | ≥ 10        | ≥ 5                 | ≥ 3            |

If the implementation work surfaces additional code paths needing coverage, raise the floor. Don't lower it.

### 6.2 Sample test scaffolding (model for new crates)

Follow the locke pattern from `tests/locke/mod.rs`:

```rust
// tests/causal/mod.rs
mod causal_proptest;
mod causal_fuzz;
mod causal_determinism;
mod causal_locke_parity;
mod propensity_score_tests;
mod iv_regression_tests;
mod double_ml_tests;
```

The unit `tests` module lives inside the crate's `lib.rs`. The integration-test files in `tests/<crate>/` are each `mod.rs` siblings sharing helpers via a private `mod common;` if needed.

---

## 7. Documentation deliverables per crate

Each crate ships with the following documentation alongside the code (matching the cjc-locke pattern):

| Artifact                                                                | Where                                                | Required?     |
| ----------------------------------------------------------------------- | ---------------------------------------------------- | ------------- |
| Crate-level `lib.rs` docstring with quick-start example                 | `crates/cjc-{name}/src/lib.rs`                       | Yes           |
| ADR documenting the v0.1 architectural decisions                        | `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-XXXX *.md`      | Yes (one per crate) |
| Determinism note (seed-flow diagram, accumulator choice, tie-breaking)  | `docs/{name}/DETERMINISM.md`                         | Yes           |
| Roadmap entry in this file (move to "Done" with date when shipped)      | `CJC-Lang_Obsidian_Vault/10_Roadmap_and_Open_Questions/New Crate Stack — Cronos, Causal, Tempest.md` | Yes |
| Memory entry summarising the ship (one-line index + topic file if dense) | `~/.claude/projects/.../memory/MEMORY.md` + per-project file | Yes  |
| Blog post on `adamezzat1.github.io`                                     | `blog/posts/{name}-v0.1-released/index.qmd`          | Yes (post-ship) |
| Python bridge — `cjc-{name}-py`                                         | `python/cjc_{name}/`                                 | Deferred to v0.1.1+ |

ADR numbering proceeds from the current `ADR-0042`. Anticipated assignments (subject to revision when ADRs are actually written):
- `ADR-0043 cjc-causal v0.1 — Propensity Score, IV, Double ML`
- `ADR-0044 cjc-cronos v0.1 — ETS, ARIMA, Kalman, STL`
- `ADR-0045 cjc-tempest v0.1 — Metropolis, HMC, NUTS with byte-identical posteriors`

---

## 8. Definition of "demo-complete" per crate

Each crate is "demo-complete" — i.e., release-ready — when:

1. ✅ Clean build: `cargo build --workspace --release` succeeds.
2. ✅ All tests pass: `cargo test --workspace --release` exits 0.
3. ✅ Test minimums met (see §6.1).
4. ✅ Determinism test passes on Linux + macOS + Windows CI.
5. ✅ Crate's ADR is committed in `13_ADRs/`.
6. ✅ Determinism note in `docs/{name}/DETERMINISM.md` exists.
7. ✅ Roadmap entry moved from "📋 Planned" to "✅ Done [date]" in this file.
8. ✅ Memory index updated.
9. ✅ One worked example in the crate's `examples/` directory demonstrates the headline use case (e.g., for cjc-causal: a synthetic randomised-vs-observational study where the causal estimator recovers the true treatment effect).
10. ✅ Crate published to crates.io as a workspace-version bump.
11. ✅ Blog post drafted (not necessarily posted) for `adamezzat1.github.io`.

The Python bridge does not block the Rust release. Each crate's Python wheel is its own follow-up release.

---

## 9. Pointers to existing patterns

When in doubt, follow these patterns from already-shipped code:

| For…                                | See…                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| Crate skeleton (Cargo.toml, lib.rs) | `crates/cjc-locke/`                                                                   |
| Test directory layout               | `tests/locke/`                                                                         |
| Proptest examples                   | `tests/locke/locke_proptest.rs`                                                       |
| Bolero fuzz examples                | `tests/locke/locke_fuzz.rs`                                                           |
| Custom Detector trait pattern       | [[ADR-0041 Locke v0.8 — Custom Detector Extension Layer]] + `crates/cjc-locke/src/custom_detector.rs` |
| Working_df pattern (auto-mutate)    | [[ADR-0042 Locke v0.8 — Str-to-Float Auto-Promotion + E9070 Wiring]] + `crates/cjc-locke/src/api.rs` |
| Findings emission                   | `crates/cjc-locke/src/report.rs`                                                      |
| Wiring builtins (3-place pattern)   | [[CLAUDE]] + Phase 3c GradGraph reference in `crates/cjc-ad/src/dispatch.rs`           |
| Python bridge via PyO3              | `python/src/lib.rs` + `python/cjc_locke/__init__.py`                                  |
| ADR format                          | [[ADR-0041 Locke v0.8 — Custom Detector Extension Layer]]                              |
| Blog post format                    | `https://adamezzat1.github.io/blog/posts/locke-v0.8-lendingclub-and-the-three-gaps/`  |

---

## 10. Open questions for the session driver

These are explicit questions for the next session(s) to decide and document:

1. **cjc-causal** — should `EffectEstimate` carry the full balance report as a sub-field, or as a separate output type? Affects the JSON emit schema.
2. **cjc-causal** — should the data-quality refusal check (the `validate_then_estimate` integration with Locke) be on by default or opt-in? Recommendation: on by default; opt-out via `CausalConfig { skip_locke_validation: bool }`.
3. **cjc-cronos** — `TimeSeries<T>` wrapper or plain DataFrame? Recommendation in §3.1 is the wrapper; the next driver may disagree.
4. **cjc-cronos** — STL implementation: ship the textbook Cleveland 1990 or the Cleveland-Hyndman robust variant? Recommendation: Cleveland 1990 in v0.1, robust as a config flag in v0.2.
5. **cjc-tempest** — closure-based vs DAG-based model representation: pick one for v0.1 (recommendation: closure-based; DAG-based as a thin wrapper in the same release).
6. **cjc-tempest** — sample mass-matrix adaptation in v0.1 or defer to v0.2? Recommendation: defer (significantly increases scope; identity mass works for v0.1's headline claim).
7. **All three** — should the v0.1 release publish all three crates together as a single workspace bump (`0.1.11 → 0.1.12`) or three separate workspace bumps (`0.1.11 → 0.1.12 → 0.1.13 → 0.1.14`)? Recommendation: separate bumps; one workspace publish per crate ship aligns the version log with the actual release timeline.

---

## 11. Final reminder for the session driver

Re-read the [[CLAUDE|Prime Directives]] before opening any source file:

1. Do not break the compiler pipeline.
2. Do not introduce hidden allocations or GC usage in NoGC-verified paths.
3. Maintain deterministic execution — same seed = bit-identical output.
4. Preserve backward compatibility unless explicitly impossible.
5. Never silently refactor unrelated systems — scope changes to the feature being implemented.
6. Language primitives must stay minimal — higher-level functionality belongs in libraries.
7. Both executors must agree — every feature must work in `cjc-eval` AND `cjc-mir-exec` (irrelevant for these crates if v0.1 is library-only and not exposed as cjcl builtins).

For these crates specifically, directives 3 and 5 are the most actively load-bearing. The determinism story is what makes each crate uniquely CJC-shaped; if you sacrifice it for a perceived performance win, you have given up the publishable claim. Don't.
