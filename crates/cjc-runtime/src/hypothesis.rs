//! Statistical hypothesis tests — t-test (one-sample, two-sample, paired),
//! chi-squared goodness-of-fit, ANOVA, F-test.
//!
//! # Determinism Contract
//! All tests are deterministic — same input => identical results.
//! Uses Kahan summation for all reductions.

use cjc_repro::KahanAccumulatorF64;
use crate::distributions::{t_cdf, chi2_cdf, f_cdf, normal_cdf};
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
// Phase B5: Weighted Least Squares
// ---------------------------------------------------------------------------

/// Weighted least squares regression.
/// Transforms to OLS: X_w = W^{1/2} * X, y_w = W^{1/2} * y.
/// Then applies standard QR-based least squares via lm().
pub fn wls(
    x_flat: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
) -> Result<LmResult, String> {
    if x_flat.len() != n * p {
        return Err(format!("wls: x_matrix size {} != n*p = {}", x_flat.len(), n * p));
    }
    if y.len() != n || weights.len() != n {
        return Err("wls: y and weights must have length n".into());
    }
    for (i, &w) in weights.iter().enumerate() {
        if w <= 0.0 {
            return Err(format!("wls: weight[{i}] = {w} must be positive"));
        }
    }
    // Transform: multiply each row by sqrt(weight)
    let mut x_w = vec![0.0; n * p];
    let mut y_w = vec![0.0; n];
    for i in 0..n {
        let sw = weights[i].sqrt();
        y_w[i] = y[i] * sw;
        for j in 0..p {
            x_w[i * p + j] = x_flat[i * p + j] * sw;
        }
    }
    lm(&x_w, &y_w, n, p)
}

// ---------------------------------------------------------------------------
// Phase B7: Non-parametric tests & multiple comparisons
// ---------------------------------------------------------------------------

/// Tukey HSD pairwise comparison result.
#[derive(Debug, Clone)]
pub struct TukeyHsdPair {
    pub group_i: usize,
    pub group_j: usize,
    pub mean_diff: f64,
    pub se: f64,
    pub q_statistic: f64,
    pub p_value: f64,
}

/// Tukey HSD post-hoc test after one-way ANOVA.
pub fn tukey_hsd(groups: &[&[f64]]) -> Result<Vec<TukeyHsdPair>, String> {
    if groups.len() < 2 {
        return Err("tukey_hsd: need at least 2 groups".into());
    }
    let k = groups.len();
    // Compute MSW (mean square within)
    let mut n_total = 0usize;
    let mut means = Vec::with_capacity(k);
    let mut ssw = KahanAccumulatorF64::new();
    for g in groups {
        if g.is_empty() { return Err("tukey_hsd: empty group".into()); }
        n_total += g.len();
        let mut acc = KahanAccumulatorF64::new();
        for &x in *g { acc.add(x); }
        let m = acc.finalize() / g.len() as f64;
        means.push(m);
        for &x in *g {
            let d = x - m;
            ssw.add(d * d);
        }
    }
    let df_w = (n_total - k) as f64;
    if df_w <= 0.0 { return Err("tukey_hsd: not enough degrees of freedom".into()); }
    let msw = ssw.finalize() / df_w;

    let mut results = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let ni = groups[i].len() as f64;
            let nj = groups[j].len() as f64;
            let se = (msw * 0.5 * (1.0 / ni + 1.0 / nj)).sqrt();
            let mean_diff = means[i] - means[j];
            let q = mean_diff.abs() / se;
            // Approximate p-value using normal approximation for studentized range
            // p ≈ k * (k-1) * (1 - Φ(q / sqrt(2)))
            let raw_p = (k as f64) * ((k - 1) as f64) * (1.0 - normal_cdf(q / 2.0_f64.sqrt()));
            let p = raw_p.min(1.0).max(0.0);
            results.push(TukeyHsdPair { group_i: i, group_j: j, mean_diff, se, q_statistic: q, p_value: p });
        }
    }
    Ok(results)
}

/// Mann-Whitney U test result.
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    pub u_statistic: f64,
    pub z_score: f64,
    pub p_value: f64,
}

/// Mann-Whitney U test (Wilcoxon rank-sum test).
pub fn mann_whitney(x: &[f64], y: &[f64]) -> Result<MannWhitneyResult, String> {
    if x.is_empty() || y.is_empty() {
        return Err("mann_whitney: both groups must be non-empty".into());
    }
    let n1 = x.len();
    let n2 = y.len();
    let n = n1 + n2;

    // Combine and rank with stable index tie-breaking
    let mut combined: Vec<(f64, usize, usize)> = Vec::with_capacity(n);
    for (i, &v) in x.iter().enumerate() {
        combined.push((v, 0, i)); // group 0
    }
    for (i, &v) in y.iter().enumerate() {
        combined.push((v, 1, i)); // group 1
    }
    combined.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    // Assign average ranks
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && combined[j].0.to_bits() == combined[i].0.to_bits() {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // R1 = sum of ranks for group 0
    let mut r1 = KahanAccumulatorF64::new();
    for (idx, &(_, group, _)) in combined.iter().enumerate() {
        if group == 0 { r1.add(ranks[idx]); }
    }
    let r1_val = r1.finalize();
    let u1 = r1_val - (n1 as f64 * (n1 as f64 + 1.0)) / 2.0;
    let u2 = (n1 as f64) * (n2 as f64) - u1;
    let u = u1.min(u2);

    let mu = (n1 as f64 * n2 as f64) / 2.0;
    let sigma = ((n1 as f64 * n2 as f64 * (n as f64 + 1.0)) / 12.0).sqrt();
    let z = if sigma > 0.0 { (u - mu) / sigma } else { 0.0 };
    let p = 2.0 * (1.0 - normal_cdf(z.abs()));

    Ok(MannWhitneyResult { u_statistic: u, z_score: z, p_value: p })
}

/// Kruskal-Wallis H test result.
#[derive(Debug, Clone)]
pub struct KruskalWallisResult {
    pub h_statistic: f64,
    pub p_value: f64,
    pub df: f64,
}

/// Kruskal-Wallis H test: non-parametric one-way ANOVA on ranks.
pub fn kruskal_wallis(groups: &[&[f64]]) -> Result<KruskalWallisResult, String> {
    if groups.len() < 2 {
        return Err("kruskal_wallis: need at least 2 groups".into());
    }
    let k = groups.len();
    let mut all: Vec<(f64, usize, usize)> = Vec::new();
    let mut group_sizes = Vec::with_capacity(k);
    for (gi, g) in groups.iter().enumerate() {
        if g.is_empty() { return Err("kruskal_wallis: empty group".into()); }
        group_sizes.push(g.len());
        for (i, &v) in g.iter().enumerate() {
            all.push((v, gi, i));
        }
    }
    let n = all.len();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    // Assign average ranks
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && all[j].0.to_bits() == all[i].0.to_bits() { j += 1; }
        let avg = (i + 1 + j) as f64 / 2.0;
        for idx in i..j { ranks[idx] = avg; }
        i = j;
    }

    // Sum of ranks per group
    let mut rank_sums = vec![KahanAccumulatorF64::new(); k];
    for (idx, &(_, gi, _)) in all.iter().enumerate() {
        rank_sums[gi].add(ranks[idx]);
    }

    let nf = n as f64;
    let mut h_acc = KahanAccumulatorF64::new();
    for (gi, acc) in rank_sums.iter().enumerate() {
        let ri = acc.clone().finalize();
        let ni = group_sizes[gi] as f64;
        h_acc.add(ri * ri / ni);
    }
    let h = (12.0 / (nf * (nf + 1.0))) * h_acc.finalize() - 3.0 * (nf + 1.0);
    let df = (k - 1) as f64;
    let p = 1.0 - chi2_cdf(h, df);

    Ok(KruskalWallisResult { h_statistic: h, p_value: p, df })
}

/// Wilcoxon signed-rank test result.
#[derive(Debug, Clone)]
pub struct WilcoxonResult {
    pub w_statistic: f64,
    pub z_score: f64,
    pub p_value: f64,
}

/// Wilcoxon signed-rank test for paired data.
pub fn wilcoxon_signed_rank(x: &[f64], y: &[f64]) -> Result<WilcoxonResult, String> {
    if x.len() != y.len() {
        return Err("wilcoxon_signed_rank: x and y must have same length".into());
    }
    if x.len() < 2 {
        return Err("wilcoxon_signed_rank: need at least 2 observations".into());
    }

    // Compute differences, remove zeros
    let mut diffs: Vec<(f64, usize)> = Vec::new(); // (abs_diff, original_index)
    let mut signs: Vec<f64> = Vec::new();
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        let d = xi - yi;
        if d.abs() > 1e-15 {
            diffs.push((d.abs(), i));
            signs.push(if d > 0.0 { 1.0 } else { -1.0 });
        }
    }
    let nr = diffs.len();
    if nr == 0 {
        return Ok(WilcoxonResult { w_statistic: 0.0, z_score: 0.0, p_value: 1.0 });
    }

    // Sort by absolute difference (stable)
    let mut indexed: Vec<(usize, f64)> = diffs.iter().enumerate().map(|(i, &(v, _))| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

    // Average ranks
    let mut ranks = vec![0.0; nr];
    let mut i = 0;
    while i < nr {
        let mut j = i;
        while j < nr && indexed[j].1.to_bits() == indexed[i].1.to_bits() { j += 1; }
        let avg = (i + 1 + j) as f64 / 2.0;
        for k in i..j { ranks[indexed[k].0] = avg; }
        i = j;
    }

    let mut w_plus = KahanAccumulatorF64::new();
    let mut w_minus = KahanAccumulatorF64::new();
    for (i, &s) in signs.iter().enumerate() {
        if s > 0.0 { w_plus.add(ranks[i]); } else { w_minus.add(ranks[i]); }
    }
    let wp = w_plus.finalize();
    let wm = w_minus.finalize();
    let w = wp.min(wm);

    let nf = nr as f64;
    let mu = nf * (nf + 1.0) / 4.0;
    let sigma = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0).sqrt();
    let z = if sigma > 0.0 { (w - mu) / sigma } else { 0.0 };
    let p = 2.0 * (1.0 - normal_cdf(z.abs()));

    Ok(WilcoxonResult { w_statistic: w, z_score: z, p_value: p })
}

/// Bonferroni correction: adjusted_p[i] = min(p[i] * m, 1.0).
pub fn bonferroni(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len() as f64;
    p_values.iter().map(|&p| (p * m).min(1.0)).collect()
}

/// Benjamini-Hochberg FDR correction.
pub fn fdr_bh(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 { return vec![]; }

    // Sort p-values keeping track of original index
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

    // Adjust: p_adj[i] = p[i] * m / rank
    let mut adjusted_sorted = vec![0.0; m];
    for (rank_0, &(_, p)) in indexed.iter().enumerate() {
        let rank = rank_0 + 1;
        adjusted_sorted[rank_0] = (p * m as f64 / rank as f64).min(1.0);
    }

    // Enforce monotonicity (backwards)
    for i in (0..m - 1).rev() {
        if adjusted_sorted[i] > adjusted_sorted[i + 1] {
            adjusted_sorted[i] = adjusted_sorted[i + 1];
        }
    }

    // Map back to original order
    let mut result = vec![0.0; m];
    for (rank_0, &(orig_idx, _)) in indexed.iter().enumerate() {
        result[orig_idx] = adjusted_sorted[rank_0];
    }
    result
}

/// Logistic regression result.
#[derive(Debug, Clone)]
pub struct LogisticResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub iterations: usize,
}

/// Logistic regression via IRLS.
/// x_flat: row-major n x p matrix (NO intercept column — auto-added).
/// y: binary 0/1 response.
pub fn logistic_regression(
    x_flat: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
) -> Result<LogisticResult, String> {
    if x_flat.len() != n * p {
        return Err(format!("logistic_regression: x size {} != n*p = {}", x_flat.len(), n * p));
    }
    if y.len() != n {
        return Err("logistic_regression: y must have length n".into());
    }
    let pp = p + 1; // with intercept

    // Build X with intercept column
    let mut x = vec![0.0; n * pp];
    for i in 0..n {
        x[i * pp] = 1.0; // intercept
        for j in 0..p {
            x[i * pp + j + 1] = x_flat[i * p + j];
        }
    }

    let mut beta = vec![0.0; pp];
    let max_iter = 100;
    let tol = 1e-8;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // mu = sigmoid(X * beta)
        let mut mu = vec![0.0; n];
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..pp {
                eta += x[i * pp + j] * beta[j];
            }
            mu[i] = 1.0 / (1.0 + (-eta).exp());
            // Clamp for numerical stability
            mu[i] = mu[i].max(1e-10).min(1.0 - 1e-10);
        }

        // W = diag(mu * (1 - mu))
        let w: Vec<f64> = mu.iter().map(|&m| m * (1.0 - m)).collect();

        // X^T W X (pp x pp)
        let mut xtwx = vec![0.0; pp * pp];
        for i in 0..n {
            for j in 0..pp {
                for k in 0..pp {
                    xtwx[j * pp + k] += x[i * pp + j] * w[i] * x[i * pp + k];
                }
            }
        }

        // X^T (y - mu)
        let mut grad = vec![0.0; pp];
        for i in 0..n {
            let r = y[i] - mu[i];
            for j in 0..pp {
                grad[j] += x[i * pp + j] * r;
            }
        }

        // Solve xtwx * delta = grad via Cholesky
        // Since xtwx is positive definite (for well-conditioned data)
        let delta = solve_symmetric(&xtwx, &grad, pp)?;

        // Update beta
        let mut max_change = 0.0_f64;
        for j in 0..pp {
            beta[j] += delta[j];
            max_change = max_change.max(delta[j].abs());
        }

        if max_change < tol {
            break;
        }
    }

    // Final mu for log-likelihood and standard errors
    let mut mu = vec![0.0; n];
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..pp { eta += x[i * pp + j] * beta[j]; }
        mu[i] = 1.0 / (1.0 + (-eta).exp());
        mu[i] = mu[i].max(1e-10).min(1.0 - 1e-10);
    }

    // Log-likelihood
    let mut ll = KahanAccumulatorF64::new();
    for i in 0..n {
        ll.add(y[i] * mu[i].ln() + (1.0 - y[i]) * (1.0 - mu[i]).ln());
    }
    let log_likelihood = ll.finalize();
    let aic = -2.0 * log_likelihood + 2.0 * pp as f64;

    // Standard errors from (X^T W X)^{-1}
    let w: Vec<f64> = mu.iter().map(|&m| m * (1.0 - m)).collect();
    let mut xtwx = vec![0.0; pp * pp];
    for i in 0..n {
        for j in 0..pp {
            for k in 0..pp {
                xtwx[j * pp + k] += x[i * pp + j] * w[i] * x[i * pp + k];
            }
        }
    }
    let inv = invert_symmetric(&xtwx, pp)?;
    let mut std_errors = Vec::with_capacity(pp);
    let mut z_values = Vec::with_capacity(pp);
    let mut p_values = Vec::with_capacity(pp);
    for j in 0..pp {
        let se = inv[j * pp + j].max(0.0).sqrt();
        std_errors.push(se);
        let z = if se > 0.0 { beta[j] / se } else { 0.0 };
        z_values.push(z);
        let pv = 2.0 * (1.0 - normal_cdf(z.abs()));
        p_values.push(pv);
    }

    Ok(LogisticResult {
        coefficients: beta,
        std_errors,
        z_values,
        p_values,
        log_likelihood,
        aic,
        iterations,
    })
}

/// Solve A*x = b for symmetric positive definite A (Cholesky).
fn solve_symmetric(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, String> {
    // Cholesky decomposition: A = L * L^T
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    // Fall back to regularized version
                    let mut a_reg = a.to_vec();
                    for ii in 0..n { a_reg[ii * n + ii] += 1e-6; }
                    return solve_symmetric(&a_reg, b, n);
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    // Forward substitution: L * y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i { sum += l[i * n + j] * y[j]; }
        y[i] = (b[i] - sum) / l[i * n + i];
    }
    // Backward substitution: L^T * x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n { sum += l[j * n + i] * x[j]; }
        x[i] = (y[i] - sum) / l[i * n + i];
    }
    Ok(x)
}

/// Invert symmetric positive definite matrix via Cholesky.
fn invert_symmetric(a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut inv = vec![0.0; n * n];
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        let x = solve_symmetric(a, &e, n)?;
        for row in 0..n {
            inv[row * n + col] = x[row];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Normality Tests
// ---------------------------------------------------------------------------

/// Result of a normality test.
pub struct NormalityResult {
    pub statistic: f64,
    pub p_value: f64,
}

/// Jarque-Bera normality test.
/// Tests whether data has skewness and kurtosis matching a normal distribution.
/// JB = (n/6) * (S² + (K-3)²/4)
pub fn jarque_bera(data: &[f64]) -> Result<NormalityResult, String> {
    let n = data.len();
    if n < 3 { return Err("jarque_bera: need at least 3 observations".into()); }
    let nf = n as f64;

    // Mean
    let mut acc = KahanAccumulatorF64::new();
    for &x in data { acc.add(x); }
    let mean = acc.finalize() / nf;

    // Central moments via Kahan
    let mut m2_acc = KahanAccumulatorF64::new();
    let mut m3_acc = KahanAccumulatorF64::new();
    let mut m4_acc = KahanAccumulatorF64::new();
    for &x in data {
        let d = x - mean;
        let d2 = d * d;
        m2_acc.add(d2);
        m3_acc.add(d2 * d);
        m4_acc.add(d2 * d2);
    }
    let m2 = m2_acc.finalize() / nf;
    let m3 = m3_acc.finalize() / nf;
    let m4 = m4_acc.finalize() / nf;

    if m2 == 0.0 { return Err("jarque_bera: zero variance".into()); }

    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2);

    let jb = (nf / 6.0) * (skewness * skewness + (kurtosis - 3.0).powi(2) / 4.0);

    // p-value from chi-squared distribution with 2 degrees of freedom
    // P(X > jb) = exp(-jb/2) for chi2(2)
    let p_value = (-jb / 2.0).exp();

    Ok(NormalityResult { statistic: jb, p_value })
}

/// Anderson-Darling test for normality.
/// Compares the empirical CDF to a normal CDF.
pub fn anderson_darling(data: &[f64]) -> Result<NormalityResult, String> {
    let n = data.len();
    if n < 8 { return Err("anderson_darling: need at least 8 observations".into()); }
    let nf = n as f64;

    // Mean and std
    let mut acc = KahanAccumulatorF64::new();
    for &x in data { acc.add(x); }
    let mean = acc.finalize() / nf;
    let mut var_acc = KahanAccumulatorF64::new();
    for &x in data { let d = x - mean; var_acc.add(d * d); }
    let std = (var_acc.finalize() / (nf - 1.0)).sqrt();
    if std == 0.0 { return Err("anderson_darling: zero standard deviation".into()); }

    // Sort and standardize
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let z: Vec<f64> = sorted.iter().map(|&x| (x - mean) / std).collect();

    // A² statistic
    let mut a2_acc = KahanAccumulatorF64::new();
    for i in 0..n {
        let phi_zi = normal_cdf(z[i]);
        let phi_zn = normal_cdf(z[n - 1 - i]);
        // Clamp to avoid log(0)
        let p1 = phi_zi.max(1e-15).min(1.0 - 1e-15);
        let p2 = phi_zn.max(1e-15).min(1.0 - 1e-15);
        let term = (2.0 * (i as f64) + 1.0) * (p1.ln() + (1.0 - p2).ln());
        a2_acc.add(term);
    }
    let a2 = -nf - a2_acc.finalize() / nf;

    // Adjusted statistic
    let a2_star = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));

    // Approximate p-value using D'Agostino & Stephens (1986) table
    let p_value = if a2_star >= 1.0359 { 0.0 }      // < 0.005
        else if a2_star >= 0.8737 { 0.01 }
        else if a2_star >= 0.6305 { 0.025 }
        else if a2_star >= 0.5091 { 0.05 }
        else if a2_star >= 0.3565 { 0.10 }
        else if a2_star >= 0.2006 { 0.25 }
        else { 0.50 };  // > 0.25

    Ok(NormalityResult { statistic: a2_star, p_value })
}

// normal_cdf is already imported from crate::distributions at the top of this file.

/// Kolmogorov-Smirnov one-sample test for normality.
/// Compares empirical distribution to a standard normal.
pub fn ks_test_normal(data: &[f64]) -> Result<NormalityResult, String> {
    let n = data.len();
    if n < 5 { return Err("ks_test: need at least 5 observations".into()); }
    let nf = n as f64;

    // Standardize
    let mut acc = KahanAccumulatorF64::new();
    for &x in data { acc.add(x); }
    let mean = acc.finalize() / nf;
    let mut var_acc = KahanAccumulatorF64::new();
    for &x in data { let d = x - mean; var_acc.add(d * d); }
    let std = (var_acc.finalize() / (nf - 1.0)).sqrt();
    if std == 0.0 { return Err("ks_test: zero standard deviation".into()); }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut d_max = 0.0f64;
    for i in 0..n {
        let z = (sorted[i] - mean) / std;
        let f_z = normal_cdf(z);
        let d_plus = ((i + 1) as f64 / nf - f_z).abs();
        let d_minus = (f_z - i as f64 / nf).abs();
        d_max = d_max.max(d_plus).max(d_minus);
    }

    // Approximate p-value using Kolmogorov distribution
    // P(D > d) ≈ 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2k²n*d²)
    let nd2 = nf * d_max * d_max;
    let mut p_value = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * kf * kf * nd2).exp();
        p_value += term;
        if term.abs() < 1e-15 { break; }
    }
    let p_value = (2.0 * p_value).max(0.0).min(1.0);

    Ok(NormalityResult { statistic: d_max, p_value })
}

// ---------------------------------------------------------------------------
// Effect Sizes
// ---------------------------------------------------------------------------

/// Cohen's d: standardized difference between two group means.
pub fn cohens_d(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() < 2 || y.len() < 2 {
        return Err("cohens_d: need at least 2 observations per group".into());
    }

    let mean_x = kahan_mean(x);
    let mean_y = kahan_mean(y);
    let var_x = kahan_var(x, mean_x);
    let var_y = kahan_var(y, mean_y);

    let nx = x.len() as f64;
    let ny = y.len() as f64;

    // Pooled standard deviation
    let sp = (((nx - 1.0) * var_x + (ny - 1.0) * var_y) / (nx + ny - 2.0)).sqrt();
    if sp == 0.0 { return Ok(0.0); }

    Ok((mean_x - mean_y) / sp)
}

/// Eta-squared: proportion of variance explained by group membership.
/// Input: array of groups (each is a slice of f64).
pub fn eta_squared(groups: &[&[f64]]) -> Result<f64, String> {
    if groups.len() < 2 { return Err("eta_squared: need at least 2 groups".into()); }

    let mut grand_acc = KahanAccumulatorF64::new();
    let mut total_n = 0usize;
    for g in groups {
        for &x in *g { grand_acc.add(x); total_n += 1; }
    }
    if total_n == 0 { return Err("eta_squared: no observations".into()); }
    let grand_mean = grand_acc.finalize() / total_n as f64;

    // SS_between and SS_total
    let mut ss_between = KahanAccumulatorF64::new();
    let mut ss_total = KahanAccumulatorF64::new();
    for g in groups {
        let gm = kahan_mean(g);
        let ni = g.len() as f64;
        ss_between.add(ni * (gm - grand_mean).powi(2));
        for &x in *g {
            ss_total.add((x - grand_mean).powi(2));
        }
    }

    let ss_t = ss_total.finalize();
    if ss_t == 0.0 { return Ok(0.0); }

    Ok(ss_between.finalize() / ss_t)
}

/// Cramér's V: association between two categorical variables.
/// Input: contingency table as flat row-major array with dimensions r x c.
pub fn cramers_v(table: &[f64], nrows: usize, ncols: usize) -> Result<f64, String> {
    if table.len() != nrows * ncols {
        return Err(format!("cramers_v: table size {} != {}x{}", table.len(), nrows, ncols));
    }
    if nrows < 2 || ncols < 2 {
        return Err("cramers_v: need at least 2x2 table".into());
    }

    // Row sums, col sums, total
    let mut row_sums = vec![0.0; nrows];
    let mut col_sums = vec![0.0; ncols];
    let mut total = 0.0;
    for r in 0..nrows {
        for c in 0..ncols {
            let v = table[r * ncols + c];
            row_sums[r] += v;
            col_sums[c] += v;
            total += v;
        }
    }
    if total == 0.0 { return Err("cramers_v: empty table".into()); }

    // Chi-squared statistic
    let mut chi2 = KahanAccumulatorF64::new();
    for r in 0..nrows {
        for c in 0..ncols {
            let expected = row_sums[r] * col_sums[c] / total;
            if expected > 0.0 {
                let diff = table[r * ncols + c] - expected;
                chi2.add(diff * diff / expected);
            }
        }
    }

    let k = (nrows.min(ncols) - 1) as f64;
    if k == 0.0 { return Ok(0.0); }

    Ok((chi2.finalize() / (total * k)).sqrt())
}

/// Levene's test for equality of variances across groups.
pub fn levene_test(groups: &[&[f64]]) -> Result<(f64, f64), String> {
    if groups.len() < 2 { return Err("levene_test: need at least 2 groups".into()); }
    let k = groups.len();
    let mut total_n = 0usize;

    // Compute |x_ij - median_i| for each group
    let mut z_groups: Vec<Vec<f64>> = Vec::with_capacity(k);
    for g in groups {
        if g.is_empty() { return Err("levene_test: empty group".into()); }
        let mut sorted = g.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let med = if sorted.len() % 2 == 0 {
            (sorted[sorted.len()/2 - 1] + sorted[sorted.len()/2]) / 2.0
        } else {
            sorted[sorted.len()/2]
        };
        let z: Vec<f64> = g.iter().map(|&x| (x - med).abs()).collect();
        total_n += z.len();
        z_groups.push(z);
    }

    // Now do one-way ANOVA on the z values
    let z_refs: Vec<&[f64]> = z_groups.iter().map(|v| v.as_slice()).collect();
    let anova = anova_oneway(&z_refs)?;
    Ok((anova.f_statistic, anova.p_value))
}

/// Bartlett's test for equality of variances.
pub fn bartlett_test(groups: &[&[f64]]) -> Result<(f64, f64), String> {
    if groups.len() < 2 { return Err("bartlett_test: need at least 2 groups".into()); }
    let k = groups.len();

    let mut vars = Vec::with_capacity(k);
    let mut ns = Vec::with_capacity(k);
    let mut total_n = 0usize;

    for g in groups {
        let n = g.len();
        if n < 2 { return Err("bartlett_test: each group needs at least 2 observations".into()); }
        let m = kahan_mean(g);
        let v = kahan_var(g, m);
        vars.push(v);
        ns.push(n);
        total_n += n;
    }

    let nk = total_n - k; // total df
    let nkf = nk as f64;

    // Pooled variance
    let mut sp2_acc = KahanAccumulatorF64::new();
    for i in 0..k {
        sp2_acc.add((ns[i] as f64 - 1.0) * vars[i]);
    }
    let sp2 = sp2_acc.finalize() / nkf;
    if sp2 == 0.0 { return Err("bartlett_test: zero pooled variance".into()); }

    // Bartlett statistic
    let mut num_acc = KahanAccumulatorF64::new();
    let mut denom_acc = KahanAccumulatorF64::new();
    for i in 0..k {
        let ni_m1 = ns[i] as f64 - 1.0;
        num_acc.add(ni_m1 * (vars[i] / sp2).max(1e-300).ln());
        denom_acc.add(1.0 / ni_m1);
    }
    let t = nkf * sp2.ln() - num_acc.finalize();
    let c = 1.0 + (1.0 / (3.0 * (k as f64 - 1.0))) * (denom_acc.finalize() - 1.0 / nkf);
    let bartlett = t / c;

    // p-value from chi2 with k-1 df (approximate using gamma)
    let df = (k - 1) as f64;
    let p_value = chi2_survival(bartlett, df);

    Ok((bartlett, p_value))
}

// Helper: chi-squared survival function P(X > x) for given df.
// Uses the regularized incomplete gamma function approximation.
fn chi2_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 { return 1.0; }
    // For integer df, use series expansion of lower incomplete gamma
    let a = df / 2.0;
    let z = x / 2.0;
    // P(X <= x) = regularized_gamma_p(a, z)
    // P(X > x) = 1 - P(X <= x)
    1.0 - regularized_gamma_p(a, z)
}

fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction
        1.0 - gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-15 { break; }
    }
    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut f = 1e-30;
    let mut c = 1e-30;
    let mut d = 1.0 / (x + 1.0 - a);
    f = d;
    for n in 1..200 {
        let an = -(n as f64) * (n as f64 - a);
        let bn = x + 2.0 * n as f64 + 1.0 - a;
        d = bn + an * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = bn + an / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 { break; }
    }
    f * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Stirling's approximation for ln(Gamma(x)).
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation (g=7, n=9)
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        let s = std::f64::consts::PI / (std::f64::consts::PI * x).sin();
        return s.abs().ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut ag = coeffs[0];
    for i in 1..9 {
        ag += coeffs[i] / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + ag.ln()
}

// Kahan-mean helper (used by effect size functions).
fn kahan_mean(data: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for &x in data { acc.add(x); }
    acc.finalize() / data.len() as f64
}

// Kahan-variance helper (given mean).
fn kahan_var(data: &[f64], mean: f64) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for &x in data { let d = x - mean; acc.add(d * d); }
    acc.finalize() / (data.len() as f64 - 1.0)
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

    #[test]
    fn test_wls_uniform_weights() {
        // WLS with uniform weights should match OLS
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();
        let w = vec![1.0; 5];
        let r = wls(&x, &y, &w, 5, 1).unwrap();
        assert!((r.coefficients[0] - 1.0).abs() < 1e-8, "intercept = {}", r.coefficients[0]);
        assert!((r.coefficients[1] - 2.0).abs() < 1e-8, "slope = {}", r.coefficients[1]);
    }

    // --- B7: Non-parametric tests ---

    #[test]
    fn test_mann_whitney_identical() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = mann_whitney(&x, &y).unwrap();
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_mann_whitney_separated() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 11.0, 12.0, 13.0, 14.0];
        let r = mann_whitney(&x, &y).unwrap();
        assert!(r.p_value < 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_kruskal_wallis_identical() {
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = kruskal_wallis(&[&g1, &g2]).unwrap();
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_kruskal_wallis_different() {
        let g1 = [1.0, 2.0, 3.0];
        let g2 = [10.0, 11.0, 12.0];
        let g3 = [20.0, 21.0, 22.0];
        let r = kruskal_wallis(&[&g1, &g2, &g3]).unwrap();
        assert!(r.p_value < 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_wilcoxon_no_difference() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9]; // very close
        let r = wilcoxon_signed_rank(&x, &y).unwrap();
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_wilcoxon_clear_shift() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y: Vec<f64> = x.iter().map(|&v| v + 5.0).collect();
        let r = wilcoxon_signed_rank(&x, &y).unwrap();
        assert!(r.p_value < 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_bonferroni_basic() {
        let pvals = [0.01, 0.04, 0.5];
        let adj = bonferroni(&pvals);
        assert!((adj[0] - 0.03).abs() < 1e-12);
        assert!((adj[1] - 0.12).abs() < 1e-12);
        assert!((adj[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_fdr_bh_known() {
        let pvals = [0.01, 0.04, 0.03, 0.5];
        let adj = fdr_bh(&pvals);
        // Sorted: 0.01(idx0), 0.03(idx2), 0.04(idx1), 0.5(idx3)
        // Rank 1: 0.01*4/1 = 0.04
        // Rank 2: 0.03*4/2 = 0.06
        // Rank 3: 0.04*4/3 ≈ 0.0533
        // Rank 4: 0.5*4/4 = 0.5
        // Monotonicity: 0.04, 0.0533, 0.0533, 0.5
        assert!(adj[0] < 0.05, "adj[0] = {}", adj[0]);
        assert!(adj[2] < 0.07, "adj[2] = {}", adj[2]);
    }

    #[test]
    fn test_fdr_bh_preserves_order() {
        let pvals = [0.5, 0.01, 0.3];
        let adj = fdr_bh(&pvals);
        // Original p[0]=0.5 should still have the largest adjusted
        assert!(adj[1] < adj[0], "adj[0]={}, adj[1]={}", adj[0], adj[1]);
    }

    #[test]
    fn test_tukey_hsd_all_same() {
        let g1 = [5.0, 5.0, 5.0, 5.0, 5.0];
        let g2 = [5.0, 5.0, 5.0, 5.0, 5.0];
        let g3 = [5.0, 5.0, 5.0, 5.0, 5.0];
        let results = tukey_hsd(&[&g1, &g2, &g3]).unwrap();
        for pair in &results {
            assert!(pair.mean_diff.abs() < 1e-12);
        }
    }

    #[test]
    fn test_tukey_hsd_one_different() {
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [2.0, 3.0, 4.0, 5.0, 6.0];
        let g3 = [20.0, 21.0, 22.0, 23.0, 24.0];
        let results = tukey_hsd(&[&g1, &g2, &g3]).unwrap();
        // g3 comparisons should be significant
        let g3_pairs: Vec<_> = results.iter().filter(|p| p.group_i == 2 || p.group_j == 2).collect();
        for pair in &g3_pairs {
            assert!(pair.p_value < 0.05, "p = {} for ({}, {})", pair.p_value, pair.group_i, pair.group_j);
        }
    }

    #[test]
    fn test_logistic_convergence() {
        // Well-separated data should converge
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let r = logistic_regression(&x, &y, 10, 1).unwrap();
        assert!(r.iterations <= 100, "iterations = {}", r.iterations);
        // Coefficient for x should be positive (higher x → more likely y=1)
        assert!(r.coefficients[1] > 0.0, "beta_1 = {}", r.coefficients[1]);
    }

    #[test]
    fn test_b7_determinism() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [6.0, 7.0, 8.0, 9.0, 10.0];
        let r1 = mann_whitney(&x, &y).unwrap();
        let r2 = mann_whitney(&x, &y).unwrap();
        assert_eq!(r1.u_statistic.to_bits(), r2.u_statistic.to_bits());
        assert_eq!(r1.p_value.to_bits(), r2.p_value.to_bits());
    }
}
