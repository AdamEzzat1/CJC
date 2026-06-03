# ADR-0044 cjc-cronos v0.1 — ETS, ARIMA, Kalman, STL

- **Status:** Proposed (2026-06-02) — scaffolding committed, implementation pending across multiple sessions
- **Crate:** `cjc-cronos` (new)
- **Companion docs:** [[New Crate Stack — Cronos, Causal, Tempest]] (handoff §3), [[ADR-0043 cjc-causal v0.1 — Propensity Score, IV, Double ML]] (sister crate, same pattern)
- **Reserved error-code range:** **E9200..=E9299**

## Context

The [[New Crate Stack — Cronos, Causal, Tempest|handoff]] §0 frames the three new crates as the decision-layer triad following Locke v0.8: causal estimates *what would happen if I intervened?*, time-series *what will happen next?*, and probabilistic programming *how uncertain am I, and why?*. cjc-cronos is the time-series leg.

Python time-series incumbents (statsmodels, Prophet, darts) do not provide structural determinism: bootstrap intervals depend on system entropy by default, Kalman filter implementations drift to non-PSD covariance under rounding, STL convergence tests vary across `numpy` versions. cjc-cronos's claim is **byte-identical backtests across runs, library upgrades, and platforms** — same publishable pattern as cjc-causal's claim about treatment-effect identifiers.

Locke composes cleanly: `cjc-locke::drift::compare(train, test)` already exists for distributional drift; cjc-cronos is the natural producer of `test` (fit on train → forecast → drift-compare to held-out actuals).

## Decisions

### 1. New workspace crate `cjc-cronos`

Workspace member under `crates/cjc-cronos/`, mirroring the `cjc-causal` scaffolding precedent. `publish = false` until v0.1 implementation lands.

Path-deps: `cjc-data`, `cjc-repro`, `cjc-runtime`, `cjc-locke`. No external runtime dependencies.

### 2. `TimeSeries<f64>` wrapper struct (not plain DataFrame)

```rust
pub struct TimeSeries {
    time: Vec<i64>,        // epoch ms or observation counter
    values: Vec<f64>,
    frequency: Frequency,
}
```

`TimeSeries::new(time, values, frequency)` enforces monotonically-increasing time index (returns [`CronosError::UnsortedTimeIndex`] on violation). `TimeSeries::from_dataframe(df, time_col, value_col, frequency)` reads from a [`cjc_data::DataFrame`] with the same checks.

**Rejected alternative**: plain DataFrame + column-name argument to every method. The wrapper carries the determinism contract (sorted time + known frequency) into every downstream method without forcing each one to re-validate. Same dependency-direction pattern Locke used for `LockeReport`.

### 3. Closed `Frequency` enum

```rust
pub enum Frequency {
    Hourly, Daily, Weekly, Monthly, Quarterly, Annual, Irregular,
}
```

`Frequency::default_seasonal_period()` returns 24 / 7 / 52 / 12 / 4 / 1 / 0 respectively (0 is the sentinel for `Irregular` — downstream methods that need a seasonal period must error explicitly).

Closed enum (not a string) so a typo is a compile error.

### 4. `Forecast` and `BacktestReport` output contracts (to be added in implementation sessions)

```rust
pub struct Forecast {
    pub horizon: usize,
    pub point_estimates: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub confidence_level: f64,
    pub fitted_model_id: FingerprintId,
}

pub struct BacktestReport {
    pub per_horizon_mae: BTreeMap<usize, f64>,
    pub per_horizon_mape: BTreeMap<usize, f64>,
    pub per_horizon_rmse: BTreeMap<usize, f64>,
    pub fitted_model_id: FingerprintId,
    pub backtest_id: FingerprintId,
}
```

`fitted_model_id` is content-addressed over (model class, hyperparameters, training data fingerprint, seed). `backtest_id` is content-addressed over `(fitted_model_id, initial_window, step, n_steps_completed)`.

Both schemas come in their own implementation session — not scaffolded now to avoid premature commitment.

### 5. v0.1 method surface

| Method | Algorithm | Session | Key references |
| --- | --- | --- | --- |
| `ETS` | Simple / Holt / Holt-Winters with additive + multiplicative seasonality | 1 | Hyndman-Athanasopoulos OTexts; grid-search hyperparameters |
| `ARIMA / SARIMA` | Yule-Walker init + CSS MLE refinement + Hannan-Rissanen MA | 2 | Brockwell-Davis Chapter 5 |
| `Kalman filter + RTS smoother` | Joseph form forward filter; local-level, local-linear-trend, BSM | 3 | Durbin-Koopman 2012 |
| `STL` | Cleveland 1990 algorithm, loess inner loop | 1 or 2 | Cleveland-Cleveland-McRae-Terpenning JOS 1990 |

Each one composes with cjc-locke for drift detection on residuals or forecast errors.

### 6. Reserved error-code range E9200..=E9299

| Code  | Severity | Trigger |
| ----- | -------- | ------- |
| E9200 | Warning  | Non-stationarity detected (ADF p > 0.05) before ARIMA fit |
| E9201 | Notice   | Seasonal pattern detected on a non-seasonal model |
| E9202 | Warning  | Forecast confidence interval exceeds historical range (extrapolation) |
| E9203 | Error    | Kalman filter innovation variance non-PSD (numerical failure) |

Implementation sessions may add E9204..=E9299 codes as needed.

### 7. Library-only Rust API for v0.1 — no language-level builtins

Same scope discipline as cjc-causal v0.1. `.cjcl` source-level access is a v0.2 question.

## Determinism contract

All six workspace-wide rules apply unmodified:

1. **All float reductions** through `cjc_repro::KahanAccumulatorF64`. Raw `.iter().sum()` banned.
2. **All map iteration** through `BTreeMap` / `BTreeSet`. No `HashMap`.
3. **All randomness** (bootstrap intervals, if any) through `cjc_repro::Rng` (SplitMix64) with explicit seeds.
4. **No FMA**.
5. **Cross-platform parity** on Linux + macOS + Windows for the same `fitted_model_id` and `backtest_id`.
6. **STL convergence** uses `|new - old| < eps` (absolute), not signed difference.

Three cjc-cronos-specific determinism contracts:

1. **Kalman filter recursions** use the Joseph form `P = (I - K H) P (I - K H)' + K R K'` rather than the simpler `P = (I - K H) P` form. Joseph form is symmetric-positive-definite-preserving; the simpler form drifts to non-PSD under accumulated rounding. If the innovation variance loses positive-definiteness despite Joseph form, emit `E9203` and return [`CronosError::Numerical`].
2. **STL convergence test** uses `(new - old).abs() < eps` with `eps = f64::EPSILON * 1e4` (configurable). The signed-difference test is a common cross-platform inconsistency source.
3. **ARIMA F-statistic / log-likelihood** is computed via the existing `cjc_runtime::hypothesis::lm` Kahan-summed path (the same OLS primitive cjc-causal Sessions 2 and 3 used). No new OLS implementation.

## Test surface (handoff §6.1 floors)

| Bucket          | Floor | Location |
| --------------- | ----- | -------- |
| Unit            | ≥ 30  | `crates/cjc-cronos/src/lib.rs` (in-module `tests`) |
| Integration     | ≥ 15  | `tests/cronos/` (multiple files per method) |
| Proptest        | ≥ 5   | `tests/cronos/cronos_proptest.rs` |
| Bolero fuzz     | ≥ 3   | `tests/cronos/cronos_fuzz.rs` |
| Determinism     | (headline) | `tests/cronos/cronos_determinism.rs` — required on Linux + macOS + Windows |
| Locke parity    | (1)   | `tests/cronos/cronos_locke_parity.rs` |

Required proptest properties (handoff §3.6):

1. Forecast of a constant series is constant.
2. Forecast of a linear-trend series is linear (extrapolated).
3. ARIMA(0,0,0) fit returns the mean.
4. STL `trend + seasonal + residual == original` within `f64::EPSILON * n_rows`.
5. Two backtests on the same series with the same seed return byte-identical `BacktestReport.fitted_model_id`.

## Alternatives considered

### A — Embed time-series methods inside cjc-runtime

Pros: zero new crate.
Cons: pollutes the runtime crate with high-level model abstractions; bloats the surface every other crate sees; same reason Locke and Causal got their own crates. Rejected.

### B — A pure Rust library outside the workspace

Pros: clean isolation.
Cons: loses workspace `Cargo.lock` reproducibility, can't share `cjc-repro` Kahan + `cjc-runtime` `lm()` without external versioning. Same reasoning as Locke ADR-0028 §C. Rejected.

### C (chosen) — New workspace crate `cjc-cronos`, mirroring cjc-causal

Pros: clean ownership boundary, shared lockfile, mirrors the [[ADR-0043 cjc-causal v0.1 — Propensity Score, IV, Double ML|cjc-causal precedent]] for consistency. Cons: one more crate to publish.

## What's deferred to v0.2+

| Capability | Defer to | Rationale |
| --- | --- | --- |
| `auto_arima` (parameter search) | v0.2 | The grid + AIC criterion is non-trivial; v0.1 ships hand-specified `(p, d, q)`. |
| Gradient-based ETS parameter fitting | v0.2 | v0.1 uses grid search for determinism by construction; gradient fitting needs careful step-size handling. |
| Multivariate VAR / VECM | v0.2 | Substantial scope; deserves its own session. |
| Prophet-style trend-changepoint detection | indefinitely | Prophet's model is opinionated and bundling it would dilute cjc-cronos's identity. |
| Neural forecasting (DeepAR / N-BEATS / TFT) | indefinitely | Orthogonal scope; might land in a `cjc-neural-forecast` sibling crate. |
| `.cjcl` language-level builtins | v0.2 | 9+ wiring sites + parity tests; focused session. |
| Python bindings via PyO3 | v0.1.1 / v0.2 | Mirrors the cjc-locke-py pattern. |

## Scope-discipline summary

v0.1 ships *the four time-series methods econometrics + applied stats courses cover first* — ETS (smoothing), ARIMA (Box-Jenkins), Kalman (state-space), STL (decomposition) — plus the cjc-locke composition hook for drift comparison. v0.2 expands to `auto_arima`, gradient-based ETS, and VAR. v0.3 considers neural forecasting if there's demand.

Per the [[CLAUDE|Prime Directives]]: directives 3 (determinism) and 5 (no silent refactors of unrelated systems) are the most load-bearing. The byte-identical-backtest story is what makes cjc-cronos uniquely CJC-shaped; sacrificing it for a perceived performance win gives up the publishable claim.
