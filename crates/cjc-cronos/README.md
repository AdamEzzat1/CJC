# cjc-cronos — deterministic time-series forecasting for CJC-Lang

ETS, ARIMA, Kalman filter + RTS smoother, STL decomposition. Same data, same
configuration ⇒ byte-identical `Forecast.fitted_model_id` and byte-identical
backtest report. Across runs, across platforms, across library upgrades.

## Why cjc-cronos

Python time-series incumbents — statsmodels, Prophet, darts — produce
forecasts whose internal state depends on hash-ordered iteration, BLAS
implementations, or numeric optimizer state that's not stable across runs.
Refitting yesterday's "production" model on the same series this morning
often produces different forecasts.

cjc-cronos threads `(seed, data, hyperparameters)` through every numerical
step so a published `Forecast.fitted_model_id` is a content-addressed
fingerprint anyone can re-derive. The hash is over the canonical byte
representation of the inputs, the fitted parameters, and the forecast
intervals; matching identifier ⇒ matching forecast, ULP for ULP.

This matters specifically for:

- **Reproducible backtests** — re-run the same `backtest_ets(...)` 6 months
  later and get the same `BacktestReport.identifier` if and only if the
  underlying data is unchanged. Diff against the production identifier to
  detect data drift, not model drift.
- **Audit trails** — regulators asking "what model produced this forecast"
  get a 64-bit hash that resolves to one identifiable fitting.
- **Cross-machine forecast pipelines** — a forecast generated on Linux x86_64
  in CI matches byte-for-byte the same call on a macOS arm64 developer
  laptop.

## Quick start

```rust
use cjc_cronos::{Ets, EtsKind, Frequency, TimeSeries};

// Construct a TimeSeries — monotonically-increasing time, finite values.
let ts = TimeSeries::new(
    /* time = */ vec![1_700_000_000, 1_700_086_400, /* ... */ 1_700_950_400],
    /* values = */ vec![100.0, 105.3, /* ... */ 142.7],
    Frequency::Daily,
)?;

// Fit Holt-Winters with multiplicative seasonality + 30-step forecast.
let forecast = Ets::new(EtsKind::HoltWintersMultiplicative)
    .fit_and_forecast(&ts, 30)?;

println!(
    "point forecast at h=1: {} ± [{}, {}]  (model id: {})",
    forecast.point_estimates[0],
    forecast.lower_bound[0],
    forecast.upper_bound[0],
    forecast.fitted_model_id,
);
# Ok::<(), cjc_cronos::CronosError>(())
```

Two runs of the snippet above on the same `ts` produce identical
`fitted_model_id`. Run on a perturbed series and the identifier changes.

## v0.1 method surface

- **`Ets`** — Exponential smoothing state-space models. `EtsKind::Simple`
  (level only), `EtsKind::Holt` (level + trend), `EtsKind::HoltWinters*`
  (additive and multiplicative seasonality). Smoothing parameters fitted by
  deterministic grid search; gradient-based fitting deferred to v0.2.

- **`Arima`** — AR(p) + MA(q) with explicit differencing order `d`.
  Yule-Walker normal-equation initialisation, Hannan-Rissanen procedure for
  the MA stage, conditional-sum-of-squares ML refinement. Hand-specified
  `(p, d, q)` in v0.1; `auto_arima` deferred to v0.2.

- **`Kalman`** — Local-level, local-linear-trend, and Basic Structural Model
  state-space variants. Forward filter uses the Joseph form
  `P_new = (I - KH) P (I - KH)' + K R K'` for positive-semi-definite
  preservation under any floating-point regime. RTS smoother for retrospective
  state distributions. STAN-style log-likelihood for parameter fitting.

- **`Stl`** — Cleveland 1990 seasonal-trend decomposition into seasonal +
  trend + residual components. Per-iteration LOESS smoother with
  Kahan-compensated weighted sums. Convergence test on `|new - old| < ε`
  (absolute), not signed difference — order-of-magnitude stable.

- **`backtest_ets`** — Rolling-origin backtest harness producing
  [`BacktestReport`] with per-fold RMSE / MAE / MAPE and a content-addressed
  identifier covering the whole backtest configuration.

All five paths return content-addressed output. The `fitted_model_id` (for
forecasts) and report identifier (for backtests) are
`cjc_locke::id::FingerprintId` values — 64-bit SplitMix64 fingerprints over
the canonical byte representation.

## Determinism contract

1. All floating-point reductions go through `cjc_repro::KahanAccumulatorF64`.
   Raw `Vec<f64>::iter().sum()` is banned anywhere in the crate.
2. All map iteration uses `BTreeMap` / `BTreeSet`. No `HashMap`.
3. Kalman recursions use the Joseph form for numerical PSD preservation.
   This is non-negotiable — `(I - KH) P` alone can drop below PSD under
   accumulated round-off, breaking the smoother.
4. STL convergence uses absolute `|new - old| < ε`, never signed. Avoids
   sign-flip-dependent loop termination.
5. ETS grid-search ties resolve by ascending `(α, β, γ)` index order — never
   by hash order.
6. No FMA. `RUSTFLAGS` must not enable `target-feature=+fma`.

See ADR-0044 §"Determinism contract" for the per-method audit checklist.

## Composing with cjc-locke

cjc-cronos is the natural producer of a "test" distribution for downstream
drift detection: fit on train → forecast → hand the forecast plus the
held-out ground truth to `cjc_locke::drift::compare(train, test)`. Cronos
emits its own findings in the reserved `E9200..=E9299` range:

- `E9200` non-stationarity flagged by Dickey-Fuller before fitting.
- `E9201` seasonality requested on a non-seasonal `Frequency`.
- `E9202` forecast horizon outside the historical range.
- `E9203` Kalman covariance lost positive-semi-definiteness.

These attach to the `CronosError::DataQualityRefusal` variant; the caller
can match on them to decide whether to fix-or-acknowledge.

## What's deferred to v0.2

- **`auto_arima`** — Hyndman-Khandakar 2008 step-wise model selection.
  Requires careful determinism work on tie-breaking when multiple
  `(p, d, q)` produce the same AIC.
- **Gradient-based ETS fitting** — replaces the v0.1 grid search; requires
  `cjc_ad::GradGraph` integration which is queued for v0.2.
- **Multivariate state-space models** — VAR, VECM. Joseph-form Kalman
  extends naturally; the missing piece is determinism-clean Cholesky.
- **Prophet-style structural decomposition** — changepoint detection,
  holiday effects. Out of scope until the v0.1 surface is field-validated.
- **Neural forecasting** (DeepAR, N-BEATS, TFT). Will compose with
  `cjc_ad::GradGraph` once both crates are at the version that supports it.

## See also

- ADR-0044 design doc:
  [`CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0044 cjc-cronos v0.1 — ETS, ARIMA, Kalman, STL.md`](https://github.com/AdamEzzat1/CJC/blob/master/CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0044%20cjc-cronos%20v0.1%20%E2%80%94%20ETS%2C%20ARIMA%2C%20Kalman%2C%20STL.md)
- Phase 2 handoff §3 — release-engineering process
- [`cjc-locke`](https://crates.io/crates/cjc-locke) — upstream drift-detection
  layer that consumes cjc-cronos forecasts as the "test" distribution
- [`cjc-causal`](https://crates.io/crates/cjc-causal) — sibling decision-layer
  crate sharing the same determinism contract and Locke composition pattern

## License

MIT
