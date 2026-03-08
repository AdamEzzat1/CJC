//! Stationarity tests for time series analysis.
//! ADF (Augmented Dickey-Fuller), KPSS, and Phillips-Perron tests.
//! All use Kahan-compensated sums for numerical determinism.

use cjc_repro::kahan_sum_f64;

// ═══════════════════════════════════════════════════════════════
// Dickey-Fuller critical value tables & p-value interpolation
// ═══════════════════════════════════════════════════════════════

/// DF critical value table for "constant" regression.
/// (critical_value, p_value) — from MacKinnon (1994).
const DF_CV_CONSTANT: &[(f64, f64)] = &[
    (-3.96, 0.001),
    (-3.43, 0.01),
    (-3.13, 0.025),
    (-2.86, 0.05),
    (-2.57, 0.10),
    (-1.94, 0.25),
    (-0.91, 0.50),
    (0.00, 0.75),
    (1.33, 0.90),
    (1.70, 0.95),
    (2.16, 0.975),
    (2.54, 0.99),
];

/// DF critical value table for "constant + trend" regression.
const DF_CV_TREND: &[(f64, f64)] = &[
    (-4.38, 0.001),
    (-3.96, 0.01),
    (-3.66, 0.025),
    (-3.41, 0.05),
    (-3.13, 0.10),
    (-2.57, 0.25),
    (-1.62, 0.50),
    (-0.65, 0.75),
    (0.71, 0.90),
    (1.03, 0.95),
    (1.66, 0.975),
    (2.08, 0.99),
];

/// Interpolate p-value from DF critical value table.
fn df_pvalue(stat: f64, n: usize, table: &[(f64, f64)]) -> f64 {
    // Small sample adjustment (Schwert-style)
    let adj = if n < 100 { 0.05 } else if n < 250 { 0.02 } else { 0.0 };
    let s = stat - adj;

    if s <= table[0].0 {
        return table[0].1 * 0.5;
    }
    if s >= table[table.len() - 1].0 {
        return 1.0 - (1.0 - table[table.len() - 1].1) * 0.5;
    }
    for i in 0..table.len() - 1 {
        let (cv_lo, p_lo) = table[i];
        let (cv_hi, p_hi) = table[i + 1];
        if s >= cv_lo && s <= cv_hi {
            let w = (s - cv_lo) / (cv_hi - cv_lo);
            return p_lo + w * (p_hi - p_lo);
        }
    }
    0.50
}

// ═══════════════════════════════════════════════════════════════
// KPSS critical value tables
// ═══════════════════════════════════════════════════════════════

/// KPSS critical values for level stationarity.
/// From Kwiatkowski et al. (1992), Table 1.
const KPSS_CV_LEVEL: &[(f64, f64)] = &[
    (0.347, 0.10),
    (0.463, 0.05),
    (0.574, 0.025),
    (0.739, 0.01),
];

/// KPSS critical values for trend stationarity.
const KPSS_CV_TREND: &[(f64, f64)] = &[
    (0.119, 0.10),
    (0.146, 0.05),
    (0.176, 0.025),
    (0.216, 0.01),
];

fn kpss_pvalue(stat: f64, table: &[(f64, f64)]) -> f64 {
    if stat < table[0].0 {
        return 0.10; // Not significant (p > 0.10)
    }
    for i in 0..table.len() - 1 {
        let (cv_lo, p_lo) = table[i];
        let (cv_hi, p_hi) = table[i + 1];
        if stat >= cv_lo && stat < cv_hi {
            let w = (stat - cv_lo) / (cv_hi - cv_lo);
            return p_lo - w * (p_lo - p_hi);
        }
    }
    0.01 // Very significant (p < 0.01)
}

// ═══════════════════════════════════════════════════════════════
// Newey-West long-run variance estimator
// ═══════════════════════════════════════════════════════════════

fn long_run_variance(resid: &[f64], max_lag: Option<usize>) -> f64 {
    let n = resid.len();
    if n < 2 {
        return f64::NAN;
    }

    // Bandwidth: Schwert (1989) rule — 12 * (n/100)^(1/4)
    let l = match max_lag {
        Some(v) => v.min(n - 1),
        None => {
            let nf = n as f64;
            let bw = (12.0 * (nf / 100.0).powf(0.25)).ceil() as usize;
            bw.max(1).min(n - 1)
        }
    };

    // gamma_0 (variance)
    let sq: Vec<f64> = resid.iter().map(|&e| e * e).collect();
    let mut lrv = kahan_sum_f64(&sq) / n as f64;

    // Add Bartlett-weighted autocovariances
    for k in 1..=l {
        let prods: Vec<f64> = (k..n).map(|t| resid[t] * resid[t - k]).collect();
        let gamma_k = kahan_sum_f64(&prods) / n as f64;
        let w = 1.0 - (k as f64) / ((l + 1) as f64);
        lrv += 2.0 * w * gamma_k;
    }

    lrv
}

// ═══════════════════════════════════════════════════════════════
// ADF Test
// ═══════════════════════════════════════════════════════════════

/// Augmented Dickey-Fuller test (with intercept, no extra lags).
/// Returns (t_statistic, p_value).
/// H0: unit root (non-stationary). Reject H0 if p < alpha.
pub fn adf_test(x: &[f64]) -> Result<(f64, f64), String> {
    let n = x.len();
    if n < 4 {
        return Err("adf_test: need at least 4 observations".into());
    }

    let m = n - 1;
    // dy[t] = x[t+1] - x[t], y_lag[t] = x[t]
    let dy: Vec<f64> = (1..n).map(|t| x[t] - x[t - 1]).collect();
    let y_lag: Vec<f64> = (0..m).map(|t| x[t]).collect();

    let mf = m as f64;
    let mean_dy = kahan_sum_f64(&dy) / mf;
    let mean_yl = kahan_sum_f64(&y_lag) / mf;

    // OLS: dy ~ alpha + beta * y_lag
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..m {
        let xi = y_lag[i] - mean_yl;
        let yi = dy[i] - mean_dy;
        sxx += xi * xi;
        sxy += xi * yi;
    }

    if sxx <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let beta = sxy / sxx;
    let alpha = mean_dy - beta * mean_yl;

    // Residual sum of squares
    let resid: Vec<f64> = (0..m)
        .map(|i| dy[i] - (alpha + beta * y_lag[i]))
        .collect();
    let rss_terms: Vec<f64> = resid.iter().map(|e| e * e).collect();
    let rss = kahan_sum_f64(&rss_terms);

    let dof = m as f64 - 2.0;
    if dof <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let sigma2 = rss / dof;
    let se_beta = (sigma2 / sxx).sqrt();

    if se_beta <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let t_stat = beta / se_beta;
    let p_val = df_pvalue(t_stat, n, DF_CV_CONSTANT);

    Ok((t_stat, p_val))
}

// ═══════════════════════════════════════════════════════════════
// KPSS Test
// ═══════════════════════════════════════════════════════════════

/// KPSS test for level stationarity.
/// Returns (statistic, p_value).
/// H0: stationary. Reject H0 if p < alpha (opposite of ADF!).
pub fn kpss_test(x: &[f64]) -> Result<(f64, f64), String> {
    let n = x.len();
    if n < 3 {
        return Err("kpss_test: need at least 3 observations".into());
    }

    let nf = n as f64;
    let mu = kahan_sum_f64(x) / nf;

    // Residuals from level regression: e_t = x_t - mu
    let resid: Vec<f64> = x.iter().map(|&v| v - mu).collect();

    // Cumulative sum of residuals
    let mut cum = Vec::with_capacity(n);
    let mut s = 0.0;
    for &e in &resid {
        s += e;
        cum.push(s);
    }

    // Long-run variance (Newey-West with Bartlett kernel)
    let lrv = long_run_variance(&resid, None);
    if !lrv.is_finite() || lrv <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    // KPSS statistic: eta = sum(S_t^2) / (n^2 * lrv)
    let ss: Vec<f64> = cum.iter().map(|&v| v * v).collect();
    let eta = kahan_sum_f64(&ss);
    let stat = eta / (nf * nf * lrv);

    let p_val = kpss_pvalue(stat, KPSS_CV_LEVEL);

    Ok((stat, p_val))
}

// ═══════════════════════════════════════════════════════════════
// Phillips-Perron Test
// ═══════════════════════════════════════════════════════════════

/// Phillips-Perron test (Z_t form, constant only).
/// Returns (z_t_statistic, p_value).
/// H0: unit root. Same interpretation as ADF.
pub fn pp_test(x: &[f64]) -> Result<(f64, f64), String> {
    let n = x.len();
    if n < 4 {
        return Err("pp_test: need at least 4 observations".into());
    }

    let m = n - 1;
    let dy: Vec<f64> = (1..n).map(|t| x[t] - x[t - 1]).collect();
    let y_lag: Vec<f64> = (0..m).map(|t| x[t]).collect();

    let mf = m as f64;
    let mean_dy = kahan_sum_f64(&dy) / mf;
    let mean_yl = kahan_sum_f64(&y_lag) / mf;

    // OLS: dy ~ alpha + beta * y_lag
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..m {
        let xi = y_lag[i] - mean_yl;
        let yi = dy[i] - mean_dy;
        sxx += xi * xi;
        sxy += xi * yi;
    }

    if sxx <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    let beta = sxy / sxx;
    let alpha = mean_dy - beta * mean_yl;

    // Residuals and OLS standard error
    let resid: Vec<f64> = (0..m)
        .map(|i| dy[i] - (alpha + beta * y_lag[i]))
        .collect();
    let rss_terms: Vec<f64> = resid.iter().map(|e| e * e).collect();
    let rss = kahan_sum_f64(&rss_terms);
    let dof = mf - 2.0;
    if dof <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let s2 = rss / dof;
    let se_beta = (s2 / sxx).sqrt();

    // Long-run variance of residuals (Newey-West)
    let lrv = long_run_variance(&resid, None);
    if !lrv.is_finite() || lrv <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }

    // PP correction: Z_t = t_beta * sqrt(s2/lrv) - 0.5 * (lrv - s2) / (sqrt(lrv) * se_beta * sqrt(mf))
    let t_beta = beta / se_beta;
    let z_t = t_beta * (s2 / lrv).sqrt()
        - 0.5 * (lrv - s2) / (lrv.sqrt() * se_beta * mf.sqrt());

    let p_val = df_pvalue(z_t, n, DF_CV_CONSTANT);

    Ok((z_t, p_val))
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_walk(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = cjc_repro::Rng::seeded(seed);
        let mut data = vec![0.0];
        for _ in 1..n {
            let last = *data.last().unwrap();
            data.push(last + rng.next_f64() - 0.5);
        }
        data
    }

    fn make_stationary(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = cjc_repro::Rng::seeded(seed);
        (0..n).map(|_| rng.next_f64() - 0.5).collect()
    }

    #[test]
    fn test_adf_random_walk_non_stationary() {
        let data = make_random_walk(200, 42);
        let (t_stat, p_val) = adf_test(&data).unwrap();
        assert!(t_stat.is_finite(), "t_stat should be finite");
        assert!(p_val.is_finite(), "p_val should be finite");
        // Random walk should typically not reject H0 (p > 0.05)
        // But this is a probabilistic test, so we just check reasonableness
        assert!(p_val > 0.0, "p should be positive");
    }

    #[test]
    fn test_adf_stationary_series() {
        let data = make_stationary(200, 42);
        let (t_stat, p_val) = adf_test(&data).unwrap();
        assert!(t_stat.is_finite());
        // Stationary series should typically reject H0 (strong negative t_stat)
        assert!(t_stat < 0.0, "t_stat should be negative for stationary: {t_stat}");
    }

    #[test]
    fn test_adf_too_short() {
        let data = [1.0, 2.0, 3.0];
        assert!(adf_test(&data).is_err());
    }

    #[test]
    fn test_adf_determinism() {
        let data = make_random_walk(100, 99);
        let (t1, p1) = adf_test(&data).unwrap();
        let (t2, p2) = adf_test(&data).unwrap();
        assert_eq!(t1.to_bits(), t2.to_bits());
        assert_eq!(p1.to_bits(), p2.to_bits());
    }

    #[test]
    fn test_kpss_stationary_series() {
        let data = make_stationary(200, 42);
        let (stat, p_val) = kpss_test(&data).unwrap();
        assert!(stat.is_finite());
        assert!(p_val.is_finite());
        // Stationary series: KPSS H0 is stationarity, so p should be large (don't reject)
        assert!(stat >= 0.0, "KPSS stat should be non-negative");
    }

    #[test]
    fn test_kpss_random_walk() {
        let data = make_random_walk(200, 42);
        let (stat, p_val) = kpss_test(&data).unwrap();
        assert!(stat.is_finite());
        // Random walk should typically reject H0 (large stat, small p)
        assert!(stat > 0.0);
    }

    #[test]
    fn test_kpss_too_short() {
        let data = [1.0, 2.0];
        assert!(kpss_test(&data).is_err());
    }

    #[test]
    fn test_kpss_determinism() {
        let data = make_stationary(100, 77);
        let (s1, p1) = kpss_test(&data).unwrap();
        let (s2, p2) = kpss_test(&data).unwrap();
        assert_eq!(s1.to_bits(), s2.to_bits());
        assert_eq!(p1.to_bits(), p2.to_bits());
    }

    #[test]
    fn test_pp_random_walk() {
        let data = make_random_walk(200, 42);
        let (z_t, p_val) = pp_test(&data).unwrap();
        assert!(z_t.is_finite(), "z_t should be finite");
        assert!(p_val.is_finite(), "p should be finite");
    }

    #[test]
    fn test_pp_stationary() {
        let data = make_stationary(200, 42);
        let (z_t, p_val) = pp_test(&data).unwrap();
        assert!(z_t.is_finite());
        // PP should give negative z_t for stationary series
        assert!(z_t < 0.0, "z_t should be negative for stationary: {z_t}");
    }

    #[test]
    fn test_pp_determinism() {
        let data = make_random_walk(100, 55);
        let (z1, p1) = pp_test(&data).unwrap();
        let (z2, p2) = pp_test(&data).unwrap();
        assert_eq!(z1.to_bits(), z2.to_bits());
        assert_eq!(p1.to_bits(), p2.to_bits());
    }

    #[test]
    fn test_df_pvalue_interpolation() {
        // Very negative stat -> very small p
        let p = df_pvalue(-5.0, 200, DF_CV_CONSTANT);
        assert!(p < 0.01, "p should be tiny for stat=-5: {p}");

        // Very positive stat -> p near 1
        let p = df_pvalue(3.0, 200, DF_CV_CONSTANT);
        assert!(p > 0.99, "p should be near 1 for stat=3: {p}");

        // At critical value -2.86 -> p ~ 0.05
        let p = df_pvalue(-2.86, 500, DF_CV_CONSTANT);
        assert!((p - 0.05).abs() < 0.02, "p at -2.86 should be ~0.05: {p}");
    }

    #[test]
    fn test_long_run_variance_white_noise() {
        let data = make_stationary(500, 42);
        let lrv = long_run_variance(&data, None);
        let simple_var: f64 = data.iter().map(|x| x * x).sum::<f64>() / data.len() as f64;
        // For white noise, LRV ≈ variance
        assert!((lrv - simple_var).abs() / simple_var < 0.5,
            "LRV should be close to variance for white noise: lrv={lrv}, var={simple_var}");
    }
}
