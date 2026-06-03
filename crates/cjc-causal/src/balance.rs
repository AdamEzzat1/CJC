//! Per-covariate balance diagnostics post-match.
//!
//! Computes the standardised mean difference (SMD) and variance ratio between
//! the matched treated and matched control groups for each covariate. The
//! output [`BalanceReport`] is the diagnostic surface for whether the matching
//! actually closed the covariate gap — `|SMD| > 0.10` post-match triggers
//! Locke finding E9102.
//!
//! ## Determinism guarantees
//!
//! 1. **All reductions Kahan-summed.** Means and variances both run through
//!    [`KahanAccumulatorF64`]; no raw `.iter().sum()` anywhere.
//! 2. **Per-covariate output is `BTreeMap<String, f64>`.** Iteration order
//!    is lexicographic by name across all platforms and runs.
//! 3. **`f64::EPSILON` sentinel for zero-variance controls.** When the
//!    control-group variance is exactly zero the variance ratio is
//!    `f64::INFINITY` (treated/0); we do *not* substitute NaN — Inf is the
//!    deterministic, comparable value. `f64::INFINITY < f64::INFINITY` is
//!    `false`, so downstream `Ord` consumers must use `total_cmp`.

use crate::estimate::BalanceReport;
use cjc_repro::KahanAccumulatorF64;
use std::collections::BTreeMap;

/// Compute mean over a subset of rows using Kahan-compensated summation.
fn kahan_mean(values: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return f64::NAN;
    }
    let mut acc = KahanAccumulatorF64::new();
    for &i in indices {
        acc.add(values[i]);
    }
    acc.finalize() / indices.len() as f64
}

/// Compute unbiased sample variance (n-1 denominator) over a subset of rows.
///
/// Two-pass formulation — first pass computes the mean, second pass
/// accumulates squared deviations. We deliberately do NOT use the one-pass
/// Welford variant here because the two-pass form has tighter rounding
/// behavior on f64 ranges typical of standardised-score data.
fn kahan_var(values: &[f64], indices: &[usize], mean: f64) -> f64 {
    let n = indices.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut acc = KahanAccumulatorF64::new();
    for &i in indices {
        let d = values[i] - mean;
        acc.add(d * d);
    }
    acc.finalize() / (n - 1) as f64
}

/// Compute the per-covariate balance report over a matched-pair subset.
///
/// # Arguments
///
/// - `covariates`: `(name, full-row values)` pairs. The `full-row values` is
///   indexed by the same row indices as `treated_idx` / `control_idx`.
/// - `treated_idx`: row indices of matched treated units.
/// - `control_idx`: row indices of paired control units (same length as
///   `treated_idx`).
/// - `n_treated_unmatched`: surfaced verbatim into the report; not used here.
///
/// # Returns
///
/// A [`BalanceReport`] with `smd_post_match` and `variance_ratio_post_match`
/// populated for each covariate (BTreeMap, lexicographic order).
pub fn compute_balance(
    covariates: &[(String, Vec<f64>)],
    treated_idx: &[usize],
    control_idx: &[usize],
    n_treated_unmatched: u64,
) -> BalanceReport {
    let mut smd: BTreeMap<String, f64> = BTreeMap::new();
    let mut vr: BTreeMap<String, f64> = BTreeMap::new();

    for (name, values) in covariates {
        let mean_t = kahan_mean(values, treated_idx);
        let mean_c = kahan_mean(values, control_idx);
        let var_t = kahan_var(values, treated_idx, mean_t);
        let var_c = kahan_var(values, control_idx, mean_c);

        // Pooled SD: sqrt((var_t + var_c) / 2). When both variances are
        // exactly zero (constant covariate post-match), the SMD is 0/0 —
        // we return 0.0 because there IS no imbalance to flag.
        let pooled_var = (var_t + var_c) / 2.0;
        let smd_value = if pooled_var == 0.0 {
            0.0
        } else {
            (mean_t - mean_c) / pooled_var.sqrt()
        };
        smd.insert(name.clone(), smd_value);

        // Variance ratio: var_t / var_c. Inf when control variance is zero.
        let vr_value = if var_c == 0.0 {
            if var_t == 0.0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            var_t / var_c
        };
        vr.insert(name.clone(), vr_value);
    }

    BalanceReport {
        smd_post_match: smd,
        variance_ratio_post_match: vr,
        n_treated_unmatched,
    }
}

/// Compute the average treatment effect on the treated (ATT) over matched
/// pairs, using Kahan-compensated summation of paired outcome differences.
///
/// The ATT here is the matched-pair mean of `y[treated] - y[control]`. With
/// well-matched controls, this approximates the average treatment effect on
/// the treated population.
///
/// # Returns
///
/// `(att_point, n_pairs)`. Returns `(f64::NAN, 0)` if `treated_idx` is empty.
pub fn compute_att(
    outcomes: &[f64],
    treated_idx: &[usize],
    control_idx: &[usize],
) -> (f64, u64) {
    debug_assert_eq!(treated_idx.len(), control_idx.len(), "matched pairs must align");
    let n = treated_idx.len();
    if n == 0 {
        return (f64::NAN, 0);
    }
    let mut acc = KahanAccumulatorF64::new();
    for k in 0..n {
        acc.add(outcomes[treated_idx[k]] - outcomes[control_idx[k]]);
    }
    (acc.finalize() / n as f64, n as u64)
}

/// Compute the matched-pair-bootstrap 95% CI for the ATT.
///
/// Bootstrap resampling is *pair-level*: at each replication we draw `n`
/// pairs with replacement using the caller-supplied
/// [`cjc_repro::Rng`](cjc_repro::Rng), compute the resampled ATT, and use
/// the empirical 2.5% / 97.5% quantiles of the bootstrap distribution as
/// the CI endpoints.
///
/// # Arguments
///
/// - `outcomes` / `treated_idx` / `control_idx`: same as [`compute_att`].
/// - `rng`: caller-supplied [`cjc_repro::Rng`]; threading the seed through
///   the call site is part of the determinism contract.
/// - `n_reps`: number of bootstrap replications (e.g. 200, 1000).
/// - `confidence_level`: e.g. `0.95` for a 95% CI.
///
/// # Returns
///
/// `(ci_lower, ci_upper, std_error)` where `std_error` is the empirical
/// standard deviation of the bootstrap ATT distribution (Kahan-summed).
pub fn bootstrap_ci(
    outcomes: &[f64],
    treated_idx: &[usize],
    control_idx: &[usize],
    rng: &mut cjc_repro::Rng,
    n_reps: usize,
    confidence_level: f64,
) -> (f64, f64, f64) {
    let n = treated_idx.len();
    if n == 0 || n_reps == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut samples: Vec<f64> = Vec::with_capacity(n_reps);
    for _ in 0..n_reps {
        let mut acc = KahanAccumulatorF64::new();
        for _ in 0..n {
            // SplitMix64 uniform over [0, n).
            let k = (rng.next_u64() % n as u64) as usize;
            acc.add(outcomes[treated_idx[k]] - outcomes[control_idx[k]]);
        }
        samples.push(acc.finalize() / n as f64);
    }

    // total_cmp keeps NaN ordering stable across platforms.
    samples.sort_by(|a, b| a.total_cmp(b));

    let alpha = (1.0 - confidence_level) / 2.0;
    let lo_idx = (alpha * n_reps as f64).floor() as usize;
    let hi_idx = (((1.0 - alpha) * n_reps as f64).ceil() as usize).saturating_sub(1);
    let lo = samples[lo_idx.min(n_reps - 1)];
    let hi = samples[hi_idx.min(n_reps - 1)];

    // Bootstrap SE: sample SD of the bootstrap ATT distribution.
    let mut mean_acc = KahanAccumulatorF64::new();
    for s in &samples {
        mean_acc.add(*s);
    }
    let mean = mean_acc.finalize() / n_reps as f64;
    let mut var_acc = KahanAccumulatorF64::new();
    for s in &samples {
        let d = *s - mean;
        var_acc.add(d * d);
    }
    let se = (var_acc.finalize() / (n_reps as f64).max(1.0)).sqrt();

    (lo, hi, se)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_data_has_zero_smd() {
        // Treated and control have identical means and variances.
        let cov = vec![("x".to_string(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])];
        let treated_idx = vec![0, 1, 2];
        let control_idx = vec![3, 4, 5];
        let report = compute_balance(&cov, &treated_idx, &control_idx, 0);
        let smd = report.smd_post_match["x"];
        assert!(smd.abs() < 1e-10, "perfectly balanced data must have SMD ≈ 0, got {}", smd);
    }

    #[test]
    fn imbalanced_data_has_positive_smd() {
        // Treated mean (5+6+7)/3 = 6; control mean (1+2+3)/3 = 2. Positive SMD expected.
        let cov = vec![("x".to_string(), vec![5.0, 6.0, 7.0, 1.0, 2.0, 3.0])];
        let treated_idx = vec![0, 1, 2];
        let control_idx = vec![3, 4, 5];
        let report = compute_balance(&cov, &treated_idx, &control_idx, 0);
        assert!(report.smd_post_match["x"] > 0.0, "treated > control should give positive SMD");
    }

    #[test]
    fn zero_variance_both_groups_returns_zero_smd() {
        // Treated all 5.0, control all 5.0 — no variance, no imbalance.
        let cov = vec![("x".to_string(), vec![5.0, 5.0, 5.0, 5.0])];
        let treated_idx = vec![0, 1];
        let control_idx = vec![2, 3];
        let report = compute_balance(&cov, &treated_idx, &control_idx, 0);
        assert_eq!(report.smd_post_match["x"], 0.0);
        assert_eq!(report.variance_ratio_post_match["x"], 1.0);
    }

    #[test]
    fn variance_ratio_with_zero_control_variance_is_infinite() {
        // Treated has variance, control is constant.
        let cov = vec![("x".to_string(), vec![1.0, 5.0, 3.0, 3.0])];
        let treated_idx = vec![0, 1]; // var of (1, 5) = 8.0
        let control_idx = vec![2, 3]; // var of (3, 3) = 0.0
        let report = compute_balance(&cov, &treated_idx, &control_idx, 0);
        assert_eq!(report.variance_ratio_post_match["x"], f64::INFINITY);
    }

    #[test]
    fn att_on_known_pair_set() {
        // Treated outcomes [10, 12], control outcomes [8, 10]. ATT = ((10-8) + (12-10))/2 = 2.0.
        let outcomes = vec![10.0, 12.0, 8.0, 10.0];
        let treated_idx = vec![0, 1];
        let control_idx = vec![2, 3];
        let (att, n) = compute_att(&outcomes, &treated_idx, &control_idx);
        assert!((att - 2.0).abs() < 1e-12);
        assert_eq!(n, 2);
    }

    #[test]
    fn att_with_empty_input_returns_nan() {
        let outcomes = vec![];
        let (att, n) = compute_att(&outcomes, &[], &[]);
        assert!(att.is_nan());
        assert_eq!(n, 0);
    }

    #[test]
    fn bootstrap_ci_has_lower_le_upper() {
        // Same outcomes mean ATT is well-defined; CI should bracket 2.0.
        let outcomes = vec![10.0, 12.0, 14.0, 8.0, 10.0, 12.0];
        let treated_idx = vec![0, 1, 2];
        let control_idx = vec![3, 4, 5];
        let mut rng = cjc_repro::Rng::seeded(42);
        let (lo, hi, se) = bootstrap_ci(&outcomes, &treated_idx, &control_idx, &mut rng, 100, 0.95);
        assert!(lo <= hi, "bootstrap lower {} > upper {}", lo, hi);
        assert!(se >= 0.0);
    }

    #[test]
    fn bootstrap_ci_same_seed_is_deterministic() {
        // Two runs with the same seed produce identical CI endpoints + SE.
        let outcomes = vec![10.0, 12.0, 14.0, 8.0, 10.0, 12.0];
        let treated_idx = vec![0, 1, 2];
        let control_idx = vec![3, 4, 5];
        let mut rng1 = cjc_repro::Rng::seeded(123);
        let mut rng2 = cjc_repro::Rng::seeded(123);
        let r1 = bootstrap_ci(&outcomes, &treated_idx, &control_idx, &mut rng1, 50, 0.95);
        let r2 = bootstrap_ci(&outcomes, &treated_idx, &control_idx, &mut rng2, 50, 0.95);
        assert_eq!(r1.0.to_bits(), r2.0.to_bits());
        assert_eq!(r1.1.to_bits(), r2.1.to_bits());
        assert_eq!(r1.2.to_bits(), r2.2.to_bits());
    }

    #[test]
    fn bootstrap_ci_zero_reps_returns_nan() {
        let outcomes = vec![1.0, 2.0];
        let mut rng = cjc_repro::Rng::seeded(0);
        let (lo, hi, se) = bootstrap_ci(&outcomes, &[0], &[1], &mut rng, 0, 0.95);
        assert!(lo.is_nan() && hi.is_nan() && se.is_nan());
    }

    #[test]
    fn balance_report_preserves_btreemap_iteration_order() {
        // Three covariates inserted in non-lexicographic order; output must be sorted.
        let cov = vec![
            ("zebra".to_string(), vec![1.0, 2.0, 3.0, 4.0]),
            ("apple".to_string(), vec![5.0, 6.0, 7.0, 8.0]),
            ("mango".to_string(), vec![9.0, 10.0, 11.0, 12.0]),
        ];
        let treated_idx = vec![0, 1];
        let control_idx = vec![2, 3];
        let report = compute_balance(&cov, &treated_idx, &control_idx, 0);
        let names: Vec<&str> = report.smd_post_match.keys().map(|k| k.as_str()).collect();
        assert_eq!(names, vec!["apple", "mango", "zebra"], "BTreeMap must yield lexicographic order");
    }
}
