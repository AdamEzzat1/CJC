//! Statistical hypothesis tests — t-test (one-sample, two-sample, paired),
//! chi-squared goodness-of-fit, ANOVA, F-test.
//!
//! # Determinism Contract
//! All tests are deterministic — same input => identical results.
//! Uses Kahan summation for all reductions.

use cjc_repro::KahanAccumulatorF64;
use crate::distributions::{t_cdf, chi2_cdf, f_cdf};
use crate::stats;

// ---------------------------------------------------------------------------
// T-test results
// ---------------------------------------------------------------------------

/// Result of a t-test.
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub p_value: f64,      // two-tailed
    pub df: f64,            // degrees of freedom
    pub mean: f64,
    pub se: f64,
}

/// One-sample t-test: is the mean significantly different from mu?
pub fn t_test(data: &[f64], mu: f64) -> Result<TTestResult, String> {
    if data.len() < 2 {
        return Err("t_test: need at least 2 observations".into());
    }
    let n = data.len() as f64;
    let mean = {
        let mut acc = KahanAccumulatorF64::new();
        for &x in data { acc.add(x); }
        acc.finalize() / n
    };
    let s = stats::sample_sd(data)?;
    let se = s / n.sqrt();
    let t = (mean - mu) / se;
    let df = n - 1.0;
    // Two-tailed p-value
    let p = 2.0 * (1.0 - t_cdf(t.abs(), df));
    Ok(TTestResult { t_statistic: t, p_value: p, df, mean, se })
}

/// Two-sample independent t-test (Welch's — unequal variance).
pub fn t_test_two_sample(x: &[f64], y: &[f64]) -> Result<TTestResult, String> {
    if x.len() < 2 || y.len() < 2 {
        return Err("t_test_two_sample: need at least 2 observations in each group".into());
    }
    let nx = x.len() as f64;
    let ny = y.len() as f64;
    let mean_x = {
        let mut acc = KahanAccumulatorF64::new();
        for &v in x { acc.add(v); }
        acc.finalize() / nx
    };
    let mean_y = {
        let mut acc = KahanAccumulatorF64::new();
        for &v in y { acc.add(v); }
        acc.finalize() / ny
    };
    let var_x = stats::sample_variance(x)?;
    let var_y = stats::sample_variance(y)?;
    let se = (var_x / nx + var_y / ny).sqrt();
    let t = (mean_x - mean_y) / se;
    // Welch-Satterthwaite degrees of freedom
    let num = (var_x / nx + var_y / ny).powi(2);
    let denom = (var_x / nx).powi(2) / (nx - 1.0) + (var_y / ny).powi(2) / (ny - 1.0);
    let df = num / denom;
    let p = 2.0 * (1.0 - t_cdf(t.abs(), df));
    Ok(TTestResult { t_statistic: t, p_value: p, df, mean: mean_x - mean_y, se })
}

/// Paired t-test.
pub fn t_test_paired(x: &[f64], y: &[f64]) -> Result<TTestResult, String> {
    if x.len() != y.len() {
        return Err("t_test_paired: arrays must have same length".into());
    }
    let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
    t_test(&diffs, 0.0)
}

// ---------------------------------------------------------------------------
// Chi-squared test
// ---------------------------------------------------------------------------

/// Result of a chi-squared test.
#[derive(Debug, Clone)]
pub struct ChiSquaredResult {
    pub chi2: f64,
    pub p_value: f64,
    pub df: f64,
}

/// Chi-squared goodness-of-fit test.
pub fn chi_squared_test(observed: &[f64], expected: &[f64]) -> Result<ChiSquaredResult, String> {
    if observed.len() != expected.len() {
        return Err("chi_squared_test: observed and expected must have same length".into());
    }
    if observed.is_empty() {
        return Err("chi_squared_test: empty data".into());
    }
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..observed.len() {
        if expected[i] <= 0.0 {
            return Err(format!("chi_squared_test: expected[{i}] must be > 0"));
        }
        let diff = observed[i] - expected[i];
        acc.add(diff * diff / expected[i]);
    }
    let chi2 = acc.finalize();
    let df = (observed.len() - 1) as f64;
    let p = 1.0 - chi2_cdf(chi2, df);
    Ok(ChiSquaredResult { chi2, p_value: p, df })
}

// ---------------------------------------------------------------------------
// ANOVA (Sprint 6)
// ---------------------------------------------------------------------------

/// Result of ANOVA.
#[derive(Debug, Clone)]
pub struct AnovaResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub df_between: f64,
    pub df_within: f64,
    pub ss_between: f64,
    pub ss_within: f64,
}

/// One-way ANOVA: compare means across groups.
pub fn anova_oneway(groups: &[&[f64]]) -> Result<AnovaResult, String> {
    if groups.len() < 2 {
        return Err("anova_oneway: need at least 2 groups".into());
    }
    let k = groups.len();
    let n_total: usize = groups.iter().map(|g| g.len()).sum();

    // Grand mean
    let mut grand_acc = KahanAccumulatorF64::new();
    for &g in groups {
        for &x in g {
            grand_acc.add(x);
        }
    }
    let grand_mean = grand_acc.finalize() / n_total as f64;

    // SS between and SS within
    let mut ss_between_acc = KahanAccumulatorF64::new();
    let mut ss_within_acc = KahanAccumulatorF64::new();
    for &g in groups {
        let ni = g.len() as f64;
        let mut group_acc = KahanAccumulatorF64::new();
        for &x in g { group_acc.add(x); }
        let group_mean = group_acc.finalize() / ni;
        let diff = group_mean - grand_mean;
        ss_between_acc.add(ni * diff * diff);
        for &x in g {
            let d = x - group_mean;
            ss_within_acc.add(d * d);
        }
    }
    let ss_between = ss_between_acc.finalize();
    let ss_within = ss_within_acc.finalize();
    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;

    if df_within <= 0.0 || ss_within == 0.0 {
        return Err("anova_oneway: insufficient data".into());
    }

    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f_stat = ms_between / ms_within;
    let p = 1.0 - f_cdf(f_stat, df_between, df_within);

    Ok(AnovaResult {
        f_statistic: f_stat,
        p_value: p,
        df_between,
        df_within,
        ss_between,
        ss_within,
    })
}

/// F-test for equality of variances.
pub fn f_test(x: &[f64], y: &[f64]) -> Result<(f64, f64), String> {
    let var_x = stats::sample_variance(x)?;
    let var_y = stats::sample_variance(y)?;
    let f = var_x / var_y;
    let df1 = (x.len() - 1) as f64;
    let df2 = (y.len() - 1) as f64;
    let p = if f > 1.0 {
        2.0 * (1.0 - f_cdf(f, df1, df2))
    } else {
        2.0 * f_cdf(f, df1, df2)
    };
    Ok((f, p))
}

// ---------------------------------------------------------------------------
// Linear Regression (Sprint 5)
// ---------------------------------------------------------------------------

/// Result of linear regression.
#[derive(Debug, Clone)]
pub struct LmResult {
    pub coefficients: Vec<f64>,   // [intercept, slope1, slope2, ...]
    pub std_errors: Vec<f64>,
    pub t_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub residuals: Vec<f64>,
    pub f_statistic: f64,
    pub f_p_value: f64,
}

/// Ordinary least squares regression: y = Xb + e.
/// x_matrix: flattened row-major (n x p), y: (n).
/// Adds intercept column automatically.
/// Uses QR decomposition for numerical stability.
pub fn lm(x_flat: &[f64], y: &[f64], n: usize, p: usize) -> Result<LmResult, String> {
    if x_flat.len() != n * p {
        return Err(format!("lm: x_matrix size {} != n*p = {}", x_flat.len(), n * p));
    }
    if y.len() != n {
        return Err(format!("lm: y length {} != n = {n}", y.len()));
    }
    if n <= p + 1 {
        return Err("lm: need n > p+1 for regression with intercept".into());
    }

    // Build design matrix with intercept: X_aug = [1 | X], shape (n, p+1)
    let pp = p + 1; // p+1 columns (intercept + predictors)
    let mut x_aug = vec![0.0; n * pp];
    for i in 0..n {
        x_aug[i * pp] = 1.0; // intercept
        for j in 0..p {
            x_aug[i * pp + (j + 1)] = x_flat[i * p + j];
        }
    }

    // QR decomposition of X_aug (m x pp) via Householder
    let m = n;
    let mut q_t_y = y.to_vec(); // will be overwritten with Q^T * y
    let mut r = x_aug.clone();

    // Householder QR in-place on r, accumulate Q^T * y
    for j in 0..pp {
        // Compute Householder vector for column j, rows j..m
        let mut norm_sq = 0.0;
        for i in j..m {
            norm_sq += r[i * pp + j] * r[i * pp + j];
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-15 {
            return Err("lm: rank-deficient design matrix".into());
        }
        let sign = if r[j * pp + j] >= 0.0 { 1.0 } else { -1.0 };
        let u0 = r[j * pp + j] + sign * norm;
        // v = [1, r[j+1,j]/u0, ..., r[m-1,j]/u0]
        let mut v = vec![0.0; m - j];
        v[0] = 1.0;
        for i in 1..(m - j) {
            v[i] = r[(j + i) * pp + j] / u0;
        }
        let tau = 2.0 / {
            let mut acc = KahanAccumulatorF64::new();
            for &vi in &v { acc.add(vi * vi); }
            acc.finalize()
        };

        // Apply reflection to r columns j..pp
        for col in j..pp {
            let mut dot = 0.0;
            for i in 0..v.len() {
                dot += v[i] * r[(j + i) * pp + col];
            }
            for i in 0..v.len() {
                r[(j + i) * pp + col] -= tau * dot * v[i];
            }
        }
        // Apply reflection to q_t_y
        {
            let mut dot = 0.0;
            for i in 0..v.len() {
                dot += v[i] * q_t_y[j + i];
            }
            for i in 0..v.len() {
                q_t_y[j + i] -= tau * dot * v[i];
            }
        }
    }

    // Back-substitute: R * beta = Q^T * y (upper-triangular R is in r[0..pp, 0..pp])
    let mut beta = vec![0.0; pp];
    for i in (0..pp).rev() {
        let mut s = q_t_y[i];
        for j in (i + 1)..pp {
            s -= r[i * pp + j] * beta[j];
        }
        beta[i] = s / r[i * pp + i];
    }

    // Residuals
    let mut residuals = vec![0.0; n];
    let mut ss_res_acc = KahanAccumulatorF64::new();
    for i in 0..n {
        let mut y_hat = 0.0;
        for j in 0..pp {
            y_hat += x_aug[i * pp + j] * beta[j];
        }
        residuals[i] = y[i] - y_hat;
        ss_res_acc.add(residuals[i] * residuals[i]);
    }
    let ss_res = ss_res_acc.finalize();

    // SS total
    let y_mean = {
        let mut acc = KahanAccumulatorF64::new();
        for &yi in y { acc.add(yi); }
        acc.finalize() / n as f64
    };
    let mut ss_tot_acc = KahanAccumulatorF64::new();
    for &yi in y {
        let d = yi - y_mean;
        ss_tot_acc.add(d * d);
    }
    let ss_tot = ss_tot_acc.finalize();

    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / ((n - pp) as f64);

    // Standard errors of coefficients
    let mse = ss_res / (n - pp) as f64;
    // Invert R'R for (X'X)^-1 diagonal — using R^-1
    let mut r_inv = vec![0.0; pp * pp];
    for i in 0..pp {
        r_inv[i * pp + i] = 1.0 / r[i * pp + i];
        for j in (0..i).rev() {
            let mut s = 0.0;
            for k in (j + 1)..=i {
                s += r[j * pp + k] * r_inv[k * pp + i];
            }
            r_inv[j * pp + i] = -s / r[j * pp + j];
        }
    }
    // diag((R^-1)(R^-1)^T) * mse
    let mut std_errors = Vec::with_capacity(pp);
    let mut t_values = Vec::with_capacity(pp);
    let mut p_values = Vec::with_capacity(pp);
    let df = (n - pp) as f64;
    for i in 0..pp {
        let mut diag = 0.0;
        for k in i..pp {
            diag += r_inv[i * pp + k] * r_inv[i * pp + k];
        }
        let se = (diag * mse).sqrt();
        std_errors.push(se);
        let t = if se > 0.0 { beta[i] / se } else { 0.0 };
        t_values.push(t);
        let pv = 2.0 * (1.0 - t_cdf(t.abs(), df));
        p_values.push(pv);
    }

    // F-statistic
    let ss_reg = ss_tot - ss_res;
    let df_reg = (pp - 1) as f64;
    let f_stat = if df_reg > 0.0 && mse > 0.0 {
        (ss_reg / df_reg) / mse
    } else {
        0.0
    };
    let f_p = 1.0 - f_cdf(f_stat, df_reg, df);

    Ok(LmResult {
        coefficients: beta,
        std_errors,
        t_values,
        p_values,
        r_squared,
        adj_r_squared,
        residuals,
        f_statistic: f_stat,
        f_p_value: f_p,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_test_known_mean() {
        let data = [5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9];
        let r = t_test(&data, 5.0).unwrap();
        // mean is very close to 5.0, so p should be large (non-significant)
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_t_test_shifted() {
        // data clearly different from 0
        let data = [10.1, 10.2, 10.0, 9.9, 10.3, 10.1, 10.0, 10.2];
        let r = t_test(&data, 0.0).unwrap();
        assert!(r.p_value < 0.001, "p = {}", r.p_value);
    }

    #[test]
    fn test_t_test_two_sample() {
        let x = [10.0, 11.0, 12.0, 13.0, 14.0];
        let y = [20.0, 21.0, 22.0, 23.0, 24.0];
        let r = t_test_two_sample(&x, &y).unwrap();
        assert!(r.p_value < 0.001);
    }

    #[test]
    fn test_chi_squared_uniform() {
        let observed = [20.0, 20.0, 20.0, 20.0, 20.0];
        let expected = [20.0, 20.0, 20.0, 20.0, 20.0];
        let r = chi_squared_test(&observed, &expected).unwrap();
        assert_eq!(r.chi2, 0.0);
        assert!((r.p_value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_anova_equal_groups() {
        let g1 = [5.0, 5.1, 4.9, 5.0, 5.2];
        let g2 = [5.0, 4.8, 5.1, 5.0, 4.9];
        let g3 = [5.1, 5.0, 4.9, 5.0, 5.1];
        let r = anova_oneway(&[&g1, &g2, &g3]).unwrap();
        // Groups with similar means → non-significant
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_anova_different_groups() {
        let g1 = [1.0, 2.0, 3.0, 2.0, 1.0];
        let g2 = [10.0, 11.0, 12.0, 11.0, 10.0];
        let g3 = [20.0, 21.0, 22.0, 21.0, 20.0];
        let r = anova_oneway(&[&g1, &g2, &g3]).unwrap();
        assert!(r.p_value < 0.001);
    }

    #[test]
    fn test_lm_simple() {
        // y = 2*x + 1
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 5.0, 7.0, 9.0, 11.0];
        let r = lm(&x, &y, 5, 1).unwrap();
        // intercept ≈ 1, slope ≈ 2
        assert!((r.coefficients[0] - 1.0).abs() < 1e-10, "intercept = {}", r.coefficients[0]);
        assert!((r.coefficients[1] - 2.0).abs() < 1e-10, "slope = {}", r.coefficients[1]);
        assert!((r.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_determinism() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r1 = t_test(&data, 3.0).unwrap();
        let r2 = t_test(&data, 3.0).unwrap();
        assert_eq!(r1.t_statistic.to_bits(), r2.t_statistic.to_bits());
        assert_eq!(r1.p_value.to_bits(), r2.p_value.to_bits());
    }
}
