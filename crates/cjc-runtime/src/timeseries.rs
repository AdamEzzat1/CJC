//! Time series analysis functions.
//!
//! Provides autocorrelation (ACF), partial autocorrelation (PACF),
//! exponential smoothing (EWMA, EMA), seasonal decomposition (additive
//! and multiplicative), differencing, and AR model fitting/forecasting.
//!
//! # Determinism
//!
//! All floating-point reductions use [`BinnedAccumulatorF64`] for
//! order-insensitive, bit-identical results across runs and platforms.
//! The Yule-Walker AR fit delegates to [`Tensor::solve`](crate::tensor::Tensor::solve)
//! which uses deterministic LU decomposition.

use crate::accumulator::BinnedAccumulatorF64;

// ---------------------------------------------------------------------------
// Helper: deterministic sum of a slice
// ---------------------------------------------------------------------------

/// Compute the sum of `data` using [`BinnedAccumulatorF64`] for determinism.
fn binned_sum(data: &[f64]) -> f64 {
    let mut acc = BinnedAccumulatorF64::new();
    acc.add_slice(data);
    acc.finalize()
}

// ---------------------------------------------------------------------------
// Helper: deterministic mean
// ---------------------------------------------------------------------------

/// Compute the arithmetic mean of `data` using [`BinnedAccumulatorF64`].
///
/// Returns `0.0` for empty slices.
fn binned_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    binned_sum(data) / data.len() as f64
}

// ---------------------------------------------------------------------------
// ACF — Autocorrelation function
// ---------------------------------------------------------------------------

/// Compute the autocorrelation function for lags 0..=max_lag.
///
/// Returns `Vec<f64>` of length `max_lag + 1` where `result[0] = 1.0`.
/// Uses the standard formula: ACF(k) = gamma(k) / gamma(0), where gamma(k)
/// is the autocovariance at lag k computed from the demeaned series.
pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![f64::NAN; max_lag + 1];
    }

    let mean = binned_mean(data);

    // Demeaned series
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Lag-0 autocovariance (variance)
    let sq: Vec<f64> = centered.iter().map(|&x| x * x).collect();
    let gamma0 = binned_sum(&sq);

    if gamma0 == 0.0 {
        // Constant series: lag-0 = 1.0, all others = 0.0
        let mut result = vec![0.0; max_lag + 1];
        result[0] = 1.0;
        return result;
    }

    let mut result = Vec::with_capacity(max_lag + 1);
    for k in 0..=max_lag {
        if k >= n {
            result.push(f64::NAN);
            continue;
        }
        let prods: Vec<f64> = (0..n - k).map(|t| centered[t] * centered[t + k]).collect();
        let gamma_k = binned_sum(&prods);
        result.push(gamma_k / gamma0);
    }

    result
}

// ---------------------------------------------------------------------------
// PACF — Partial autocorrelation function (Durbin-Levinson)
// ---------------------------------------------------------------------------

/// Compute the partial autocorrelation function via the Durbin-Levinson algorithm.
///
/// Returns `Vec<f64>` of length `max_lag + 1` where `result[0] = 1.0`.
pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || max_lag == 0 {
        return vec![1.0];
    }

    // First compute the ACF values we need
    let r = acf(data, max_lag);

    let mut result = vec![0.0; max_lag + 1];
    result[0] = 1.0;

    if max_lag >= n {
        // Can't compute beyond data length
        for i in n..=max_lag {
            result[i] = f64::NAN;
        }
    }

    // Durbin-Levinson recursion
    // phi[m][j] = AR coefficient at order m, index j
    // We only need two rows: current and previous.
    let effective_max = max_lag.min(n - 1);

    let mut phi_prev = vec![0.0; effective_max + 1];
    // Order 1
    phi_prev[1] = r[1];
    result[1] = r[1];

    for m in 2..=effective_max {
        // Compute phi[m][m] using Durbin-Levinson:
        // phi[m][m] = (r[m] - sum_{j=1}^{m-1} phi[m-1][j] * r[m-j]) / (1 - sum_{j=1}^{m-1} phi[m-1][j] * r[j])
        let num_terms: Vec<f64> = (1..m).map(|j| phi_prev[j] * r[m - j]).collect();
        let den_terms: Vec<f64> = (1..m).map(|j| phi_prev[j] * r[j]).collect();

        let num = r[m] - binned_sum(&num_terms);
        let den = 1.0 - binned_sum(&den_terms);

        if den.abs() < 1e-15 {
            result[m] = f64::NAN;
            break;
        }

        let phi_mm = num / den;
        result[m] = phi_mm;

        // Update phi coefficients for next iteration
        let mut phi_new = vec![0.0; effective_max + 1];
        for j in 1..m {
            phi_new[j] = phi_prev[j] - phi_mm * phi_prev[m - j];
        }
        phi_new[m] = phi_mm;
        phi_prev = phi_new;
    }

    result
}

// ---------------------------------------------------------------------------
// EWMA — Exponential weighted moving average
// ---------------------------------------------------------------------------

/// Compute the exponential weighted moving average.
///
/// `alpha` is the smoothing factor (0 < alpha <= 1).
/// Returns `Vec<f64>` of the same length as `data`.
/// The first value is `data[0]`; subsequent values are `alpha * data[i] + (1 - alpha) * ewma[i-1]`.
pub fn ewma(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);

    for i in 1..data.len() {
        let prev = result[i - 1];
        result.push(alpha * data[i] + (1.0 - alpha) * prev);
    }

    result
}

// ---------------------------------------------------------------------------
// EMA — Exponential moving average (span-based)
// ---------------------------------------------------------------------------

/// Compute the exponential moving average with span-based smoothing.
///
/// `alpha = 2 / (span + 1)`.
/// Returns `Vec<f64>` of the same length as `data`.
pub fn ema(data: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    ewma(data, alpha)
}

// ---------------------------------------------------------------------------
// Seasonal decomposition
// ---------------------------------------------------------------------------

/// Decompose a time series into trend, seasonal, and residual components.
///
/// `period`: the seasonal period (e.g., 12 for monthly data with yearly seasonality).
/// `model`: `"additive"` or `"multiplicative"`.
///
/// Returns `(trend, seasonal, residual)` each as `Vec<f64>` of the same length as `data`.
/// Boundary values where the centered moving average cannot be computed are set to `f64::NAN`.
pub fn seasonal_decompose(
    data: &[f64],
    period: usize,
    model: &str,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let n = data.len();

    if period < 2 {
        return Err("seasonal_decompose: period must be >= 2".into());
    }
    if n < 2 * period {
        return Err(format!(
            "seasonal_decompose: need at least {} observations for period {}, got {}",
            2 * period,
            period,
            n
        ));
    }
    if model != "additive" && model != "multiplicative" {
        return Err(format!(
            "seasonal_decompose: model must be \"additive\" or \"multiplicative\", got \"{}\"",
            model
        ));
    }

    let is_mult = model == "multiplicative";

    // Check for zeros/negatives in multiplicative mode
    if is_mult {
        for &v in data {
            if v <= 0.0 {
                return Err(
                    "seasonal_decompose: multiplicative model requires all positive data".into(),
                );
            }
        }
    }

    // Step 1: Centered moving average for trend extraction
    let trend = centered_moving_average(data, period);

    // Step 2: Detrend
    let mut detrended = vec![f64::NAN; n];
    for i in 0..n {
        if trend[i].is_nan() {
            continue;
        }
        if is_mult {
            if trend[i] != 0.0 {
                detrended[i] = data[i] / trend[i];
            }
        } else {
            detrended[i] = data[i] - trend[i];
        }
    }

    // Step 3: Average detrended values for each period position
    let mut seasonal = vec![0.0; n];
    let mut period_avgs = vec![0.0; period];

    for p in 0..period {
        let mut vals = Vec::new();
        let mut idx = p;
        while idx < n {
            if !detrended[idx].is_nan() {
                vals.push(detrended[idx]);
            }
            idx += period;
        }
        if !vals.is_empty() {
            period_avgs[p] = binned_mean(&vals);
        }
    }

    // Normalize seasonal component so it sums to 0 (additive) or averages to 1 (multiplicative)
    if is_mult {
        let avg = binned_mean(&period_avgs);
        if avg != 0.0 {
            for v in &mut period_avgs {
                *v /= avg;
            }
        }
    } else {
        let avg = binned_mean(&period_avgs);
        for v in &mut period_avgs {
            *v -= avg;
        }
    }

    // Tile the seasonal pattern
    for i in 0..n {
        seasonal[i] = period_avgs[i % period];
    }

    // Step 4: Residual
    let mut residual = vec![f64::NAN; n];
    for i in 0..n {
        if trend[i].is_nan() {
            continue;
        }
        if is_mult {
            if seasonal[i] != 0.0 {
                residual[i] = data[i] / (trend[i] * seasonal[i]);
            }
        } else {
            residual[i] = data[i] - trend[i] - seasonal[i];
        }
    }

    Ok((trend, seasonal, residual))
}

/// Compute the centered moving average of length `period`.
///
/// For odd periods, uses a simple symmetric window of `period` elements.
/// For even periods, applies a two-pass convolution: first a trailing
/// `period`-length MA, then averages adjacent pairs to center the result.
/// Boundary positions where the full window cannot be placed are set to
/// [`f64::NAN`].
fn centered_moving_average(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![f64::NAN; n];

    if period % 2 == 1 {
        // Odd period: simple centered MA
        let half = period / 2;
        for i in half..n.saturating_sub(half) {
            let window = &data[i - half..=i + half];
            result[i] = binned_mean(window);
        }
    } else {
        // Even period: first compute period-length MA, then average adjacent pairs
        let mut ma = vec![f64::NAN; n];
        let half = period / 2;
        // First pass: period-length trailing MA starting at index period-1
        for i in (period - 1)..n {
            let window = &data[i + 1 - period..=i];
            ma[i] = binned_mean(window);
        }
        // Second pass: center by averaging adjacent MA values
        for i in half..n.saturating_sub(half) {
            let left_idx = i + half - 1; // index in ma for trailing MA ending at i+half-1
            let right_idx = left_idx + 1;
            if left_idx < n && right_idx < n && !ma[left_idx].is_nan() && !ma[right_idx].is_nan()
            {
                result[i] = (ma[left_idx] + ma[right_idx]) / 2.0;
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Differencing
// ---------------------------------------------------------------------------

/// Difference a series: `y[i] = data[i + periods] - data[i]`.
///
/// Returns `Vec<f64>` of length `data.len() - periods`.
pub fn diff(data: &[f64], periods: usize) -> Vec<f64> {
    if periods >= data.len() {
        return Vec::new();
    }
    (periods..data.len())
        .map(|i| data[i] - data[i - periods])
        .collect()
}

// ---------------------------------------------------------------------------
// ARIMA primitives
// ---------------------------------------------------------------------------

/// ARIMA(p,d,q) differencing step.
///
/// Applies first-order differencing `d` times to produce a stationary series.
/// Returns the d-th order differenced series.
/// After one round: result\[i\] = data\[i+1\] - data\[i\], length n-1.
/// After d rounds: length n-d.
pub fn arima_diff(data: &[f64], d: usize) -> Vec<f64> {
    let mut current = data.to_vec();
    for _ in 0..d {
        if current.len() <= 1 {
            return Vec::new();
        }
        current = diff(&current, 1);
    }
    current
}

/// Fit an AR(p) model using the Yule-Walker method.
///
/// 1. Compute autocorrelation r\[0..=p\] using `acf`.
/// 2. Build the p x p Toeplitz matrix R where R\[i,j\] = r\[|i-j|\].
/// 3. Solve R * phi = r\[1..=p\] using LU decomposition (via `Tensor::solve`).
///
/// Returns the AR coefficients phi\[1..p\] as a `Vec<f64>`.
///
/// **Determinism:** ACF uses `BinnedAccumulatorF64`; solve uses deterministic LU.
pub fn ar_fit(data: &[f64], p: usize) -> Result<Vec<f64>, String> {
    if p == 0 {
        return Err("ar_fit: p must be > 0".into());
    }
    if data.len() <= p {
        return Err(format!(
            "ar_fit: need at least {} observations for AR({}), got {}",
            p + 1,
            p,
            data.len()
        ));
    }

    let r = acf(data, p);

    // Build Toeplitz matrix R: R[i][j] = r[|i-j|]
    let mut mat_data = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let lag = if i >= j { i - j } else { j - i };
            mat_data[i * p + j] = r[lag];
        }
    }

    // RHS: r[1..=p]
    let rhs: Vec<f64> = (1..=p).map(|k| r[k]).collect();

    use crate::tensor::Tensor;
    let r_matrix =
        Tensor::from_vec(mat_data, &[p, p]).map_err(|e| format!("ar_fit: {e}"))?;
    let r_vec = Tensor::from_vec(rhs, &[p]).map_err(|e| format!("ar_fit: {e}"))?;
    let phi_tensor = r_matrix.solve(&r_vec).map_err(|e| format!("ar_fit: {e}"))?;

    Ok(phi_tensor.to_vec())
}

/// AR forecast: given fitted AR coefficients and recent history, predict the
/// next `steps` values.
///
/// `coeffs`: AR coefficients \[phi_1, phi_2, ..., phi_p\] (length p).
/// `history`: recent observations, at least p values.
/// `steps`: number of future values to predict.
///
/// Each prediction is: y_hat = sum(phi_i * y\[t-i\]) for i=1..p.
/// Uses Kahan summation for determinism.
pub fn ar_forecast(coeffs: &[f64], history: &[f64], steps: usize) -> Result<Vec<f64>, String> {
    let p = coeffs.len();
    if p == 0 {
        return Err("ar_forecast: need at least one coefficient".into());
    }
    if history.len() < p {
        return Err(format!(
            "ar_forecast: need at least {} history values for AR({}), got {}",
            p,
            p,
            history.len()
        ));
    }

    // Work buffer: copy the tail of history + space for predictions
    let mut buf: Vec<f64> = history.to_vec();
    let mut predictions = Vec::with_capacity(steps);

    for _ in 0..steps {
        let n = buf.len();
        let mut acc = BinnedAccumulatorF64::new();
        for i in 0..p {
            acc.add(coeffs[i] * buf[n - 1 - i]);
        }
        let val = acc.finalize();
        predictions.push(val);
        buf.push(val);
    }

    Ok(predictions)
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- ACF tests ----------------------------------------------------------

    #[test]
    fn test_acf_constant_series() {
        let data = vec![5.0; 100];
        let result = acf(&data, 5);
        assert_eq!(result.len(), 6);
        assert_eq!(result[0], 1.0);
        for k in 1..=5 {
            assert_eq!(result[k], 0.0, "ACF at lag {} should be 0 for constant series", k);
        }
    }

    #[test]
    fn test_acf_lag_zero_is_one() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let result = acf(&data, 10);
        assert!((result[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_acf_sinusoidal_periodicity() {
        // Sine wave with period 20: ACF should show periodic pattern
        let data: Vec<f64> = (0..200)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
            .collect();
        let result = acf(&data, 25);
        // ACF at lag 20 should be close to 1.0 (same phase)
        assert!(
            result[20] > 0.9,
            "ACF at lag=period should be high, got {}",
            result[20]
        );
        // ACF at lag 10 should be close to -1.0 (half period)
        assert!(
            result[10] < -0.9,
            "ACF at lag=period/2 should be negative, got {}",
            result[10]
        );
    }

    #[test]
    fn test_acf_empty() {
        let result = acf(&[], 3);
        assert_eq!(result.len(), 4);
        assert!(result[0].is_nan());
    }

    #[test]
    fn test_acf_determinism() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1 + (i as f64).sin()).collect();
        let r1 = acf(&data, 10);
        let r2 = acf(&data, 10);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // -- PACF tests ---------------------------------------------------------

    #[test]
    fn test_pacf_lag_zero_is_one() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.3).collect();
        let result = pacf(&data, 5);
        assert!((result[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pacf_ar1_process() {
        // AR(1) process: x[t] = 0.8 * x[t-1] + noise
        // PACF should have significant value at lag 1, near zero after
        let mut data = vec![0.0; 500];
        let mut rng = cjc_repro::Rng::seeded(42);
        for t in 1..500 {
            data[t] = 0.8 * data[t - 1] + (rng.next_f64() - 0.5) * 0.1;
        }
        let result = pacf(&data, 5);
        assert!(
            result[1].abs() > 0.5,
            "PACF at lag 1 should be significant for AR(1), got {}",
            result[1]
        );
        // Lags 2+ should be much smaller
        for k in 2..=5 {
            assert!(
                result[k].abs() < 0.3,
                "PACF at lag {} should be small for AR(1), got {}",
                k,
                result[k]
            );
        }
    }

    #[test]
    fn test_pacf_determinism() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64).cos()).collect();
        let r1 = pacf(&data, 5);
        let r2 = pacf(&data, 5);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // -- EWMA tests ---------------------------------------------------------

    #[test]
    fn test_ewma_alpha_one_returns_original() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewma(&data, 1.0);
        assert_eq!(result, data);
    }

    #[test]
    fn test_ewma_alpha_zero_returns_first() {
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let result = ewma(&data, 0.0);
        // alpha=0 means ewma[i] = ewma[i-1] for all i, so all values = data[0]
        for &v in &result {
            assert_eq!(v, 10.0);
        }
    }

    #[test]
    fn test_ewma_length() {
        let data = vec![1.0, 2.0, 3.0];
        let result = ewma(&data, 0.5);
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_ewma_empty() {
        let result = ewma(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_ewma_smoothing() {
        // With alpha=0.5: ewma[0]=1, ewma[1]=0.5*3+0.5*1=2, ewma[2]=0.5*5+0.5*2=3.5
        let data = vec![1.0, 3.0, 5.0];
        let result = ewma(&data, 0.5);
        assert_eq!(result[0], 1.0);
        assert!((result[1] - 2.0).abs() < 1e-12);
        assert!((result[2] - 3.5).abs() < 1e-12);
    }

    // -- EMA tests ----------------------------------------------------------

    #[test]
    fn test_ema_span_relationship() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let span = 3;
        let alpha = 2.0 / (span as f64 + 1.0); // 0.5
        let ema_result = ema(&data, span);
        let ewma_result = ewma(&data, alpha);
        for (a, b) in ema_result.iter().zip(ewma_result.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // -- diff tests ---------------------------------------------------------

    #[test]
    fn test_diff_first_differences() {
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let result = diff(&data, 1);
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_diff_periods_two() {
        let data = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let result = diff(&data, 2);
        // result[0] = data[2] - data[0] = 3, result[1] = data[3] - data[1] = 5, result[2] = data[4] - data[2] = 7
        assert_eq!(result, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_diff_empty_when_periods_too_large() {
        let data = vec![1.0, 2.0];
        let result = diff(&data, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_diff_length() {
        let data = vec![1.0; 10];
        let result = diff(&data, 3);
        assert_eq!(result.len(), 7);
    }

    // -- seasonal_decompose tests -------------------------------------------

    #[test]
    fn test_seasonal_decompose_additive_reconstruction() {
        // Create data with known trend + seasonal + noise
        let period = 4;
        let n = 40;
        let mut data = vec![0.0; n];
        for i in 0..n {
            let trend = 10.0 + 0.5 * i as f64;
            let seasonal = [2.0, -1.0, 0.5, -1.5][i % period];
            data[i] = trend + seasonal;
        }

        let (trend, seasonal, residual) =
            seasonal_decompose(&data, period, "additive").unwrap();

        // For non-NaN positions, trend + seasonal + residual ≈ original
        for i in 0..n {
            if trend[i].is_nan() || residual[i].is_nan() {
                continue;
            }
            let reconstructed = trend[i] + seasonal[i] + residual[i];
            assert!(
                (reconstructed - data[i]).abs() < 1e-10,
                "Reconstruction failed at i={}: {} vs {}",
                i,
                reconstructed,
                data[i]
            );
        }
    }

    #[test]
    fn test_seasonal_decompose_multiplicative_reconstruction() {
        let period = 4;
        let n = 40;
        let mut data = vec![0.0; n];
        for i in 0..n {
            let trend = 100.0 + 2.0 * i as f64;
            let seasonal = [1.1, 0.9, 1.05, 0.95][i % period];
            data[i] = trend * seasonal;
        }

        let (trend, seasonal, residual) =
            seasonal_decompose(&data, period, "multiplicative").unwrap();

        for i in 0..n {
            if trend[i].is_nan() || residual[i].is_nan() {
                continue;
            }
            let reconstructed = trend[i] * seasonal[i] * residual[i];
            assert!(
                (reconstructed - data[i]).abs() < 1e-6,
                "Multiplicative reconstruction failed at i={}: {} vs {}",
                i,
                reconstructed,
                data[i]
            );
        }
    }

    #[test]
    fn test_seasonal_decompose_invalid_period() {
        let data = vec![1.0; 20];
        assert!(seasonal_decompose(&data, 1, "additive").is_err());
    }

    #[test]
    fn test_seasonal_decompose_too_short() {
        let data = vec![1.0; 5];
        assert!(seasonal_decompose(&data, 4, "additive").is_err());
    }

    #[test]
    fn test_seasonal_decompose_invalid_model() {
        let data = vec![1.0; 20];
        assert!(seasonal_decompose(&data, 4, "invalid").is_err());
    }

    #[test]
    fn test_seasonal_decompose_multiplicative_negative_data() {
        let data = vec![1.0, -1.0, 2.0, 3.0, 1.0, -1.0, 2.0, 3.0];
        assert!(seasonal_decompose(&data, 4, "multiplicative").is_err());
    }

    #[test]
    fn test_seasonal_decompose_seasonal_component_sums_to_zero() {
        let period = 4;
        let n = 40;
        let mut data = vec![0.0; n];
        for i in 0..n {
            data[i] = 10.0 + 0.5 * i as f64 + [2.0, -1.0, 0.5, -1.5][i % period];
        }

        let (_, seasonal, _) = seasonal_decompose(&data, period, "additive").unwrap();

        // One full period of seasonal component should sum to ~0
        let one_period: Vec<f64> = (0..period).map(|i| seasonal[i]).collect();
        let period_sum = binned_sum(&one_period);
        assert!(
            period_sum.abs() < 1e-10,
            "Seasonal component should sum to ~0 over one period, got {}",
            period_sum
        );
    }

    #[test]
    fn test_seasonal_decompose_determinism() {
        let period = 4;
        let n = 40;
        let data: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64).sin()).collect();

        let (t1, s1, r1) = seasonal_decompose(&data, period, "additive").unwrap();
        let (t2, s2, r2) = seasonal_decompose(&data, period, "additive").unwrap();

        for i in 0..n {
            assert_eq!(t1[i].to_bits(), t2[i].to_bits());
            assert_eq!(s1[i].to_bits(), s2[i].to_bits());
            assert_eq!(r1[i].to_bits(), r2[i].to_bits());
        }
    }
}
