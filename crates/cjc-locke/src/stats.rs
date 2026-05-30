//! Deterministic numeric helpers used by validation and drift.
//!
//! Every reduction in this module routes through `cjc_repro::Kahan*` so
//! repeated runs over the same data produce bit-identical scalars. We
//! also skip NaN by convention — NaN signals "missing" in `Column::Float`,
//! and including it in a sum would propagate NaN through every metric.

use cjc_repro::KahanAccumulatorF64;

/// Summary statistics for one numeric column.
///
/// `None` values indicate a column with zero non-NaN observations — in
/// that case mean/variance are undefined and callers must check `n_valid`.
#[derive(Clone, Debug, PartialEq)]
pub struct NumericSummary {
    pub n_total: u64,
    pub n_valid: u64,
    pub n_missing: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub variance: Option<f64>,
    pub std_dev: Option<f64>,
}

impl NumericSummary {
    pub fn empty(n_total: u64) -> Self {
        Self {
            n_total,
            n_valid: 0,
            n_missing: n_total,
            min: None,
            max: None,
            mean: None,
            variance: None,
            std_dev: None,
        }
    }
}

/// Compute a deterministic summary of an f64 slice. NaN is treated as
/// missing. Variance uses the population (N) denominator, not (N-1) —
/// Locke is a descriptive tool, not an inferential one, and we want
/// the result to be defined for n_valid == 1.
pub fn summarize_f64(values: &[f64]) -> NumericSummary {
    let n_total = values.len() as u64;
    if n_total == 0 {
        return NumericSummary::empty(0);
    }

    let mut acc = KahanAccumulatorF64::new();
    let mut n_valid: u64 = 0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_nan() {
            continue;
        }
        acc.add(v);
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        n_valid += 1;
    }

    if n_valid == 0 {
        return NumericSummary::empty(n_total);
    }

    let sum = acc.finalize();
    let mean = sum / n_valid as f64;

    // Variance: deterministic Kahan-summed (x - mean)^2.
    let mut var_acc = KahanAccumulatorF64::new();
    for &v in values {
        if v.is_nan() {
            continue;
        }
        let d = v - mean;
        var_acc.add(d * d);
    }
    let variance = var_acc.finalize() / n_valid as f64;
    let std_dev = variance.sqrt();

    NumericSummary {
        n_total,
        n_valid,
        n_missing: n_total - n_valid,
        min: Some(min),
        max: Some(max),
        mean: Some(mean),
        variance: Some(variance),
        std_dev: Some(std_dev),
    }
}

/// Same as [`summarize_f64`] but for integer columns. Integers cannot be
/// missing, so `n_missing` is always 0 (unless the caller supplies a null
/// mask via a future API — deferred to v0.2).
pub fn summarize_i64(values: &[i64]) -> NumericSummary {
    let n_total = values.len() as u64;
    if n_total == 0 {
        return NumericSummary::empty(0);
    }
    let mut acc = KahanAccumulatorF64::new();
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for &v in values {
        acc.add(v as f64);
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let n_valid = n_total;
    let mean = acc.finalize() / n_valid as f64;
    let mut var_acc = KahanAccumulatorF64::new();
    for &v in values {
        let d = v as f64 - mean;
        var_acc.add(d * d);
    }
    let variance = var_acc.finalize() / n_valid as f64;
    NumericSummary {
        n_total,
        n_valid,
        n_missing: 0,
        min: Some(min as f64),
        max: Some(max as f64),
        mean: Some(mean),
        variance: Some(variance),
        std_dev: Some(variance.sqrt()),
    }
}

/// Pearson correlation between two equal-length numeric vectors with
/// NaN-as-missing semantics. Returns `None` if fewer than 2 valid pairs
/// or if either side has zero variance.
pub fn pearson_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return None;
    }

    let mut sum_x = KahanAccumulatorF64::new();
    let mut sum_y = KahanAccumulatorF64::new();
    let mut n_pair: u64 = 0;
    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        if x.is_nan() || y.is_nan() {
            continue;
        }
        sum_x.add(x);
        sum_y.add(y);
        n_pair += 1;
    }
    if n_pair < 2 {
        return None;
    }
    let mean_x = sum_x.finalize() / n_pair as f64;
    let mean_y = sum_y.finalize() / n_pair as f64;

    let mut s_xy = KahanAccumulatorF64::new();
    let mut s_xx = KahanAccumulatorF64::new();
    let mut s_yy = KahanAccumulatorF64::new();
    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        if x.is_nan() || y.is_nan() {
            continue;
        }
        let dx = x - mean_x;
        let dy = y - mean_y;
        s_xy.add(dx * dy);
        s_xx.add(dx * dx);
        s_yy.add(dy * dy);
    }
    let denom = (s_xx.finalize() * s_yy.finalize()).sqrt();
    if denom == 0.0 || !denom.is_finite() {
        return None;
    }
    Some(s_xy.finalize() / denom)
}

/// Exact Kolmogorov–Smirnov D-statistic between two empirical CDFs.
///
/// D = sup_x |F_n(x) − F_m(x)| where F_n is the empirical CDF of `xs`
/// (NaN-filtered) and F_m of `ys`. Returns `None` if either sample has
/// fewer than 2 non-NaN observations.
///
/// Algorithm: sort each side once (stable sort), then merge-walk the two
/// sorted streams maintaining running CDF positions. The maximum
/// absolute gap is the answer. O((n + m) · log(n + m)).
///
/// Determinism: the only ordering-sensitive operation is the sort, which
/// uses `f64::total_cmp` so NaN-free inputs produce a fully-determined
/// order across platforms. Repeated calls yield bit-identical D.
pub fn ks_d_statistic(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let a = sort_filter_nan(xs);
    let b = sort_filter_nan(ys);
    ks_d_statistic_sorted(&a, &b)
}

/// Filter NaN and sort by `f64::total_cmp`. Shared helper for callers
/// that want to sort once and reuse across multiple distribution
/// metrics (e.g. KS and Wasserstein in [`drift::numeric_ks_finding`]).
pub fn sort_filter_nan(xs: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
    v.sort_by(|x, y| x.total_cmp(y));
    v
}

/// KS-D computed from pre-sorted, NaN-free inputs. Identical contract
/// to [`ks_d_statistic`] but skips the filter+sort step — callers
/// computing multiple sorted-input metrics (KS + Wasserstein) avoid
/// the double-sort cost by going through this entry point.
pub fn ks_d_statistic_sorted(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len();
    let m = b.len();
    if n < 2 || m < 2 {
        return None;
    }

    let n_f = n as f64;
    let m_f = m as f64;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d_max: f64 = 0.0;

    // v0.7+ deep-dive bug-fix (was CRITICAL): the previous implementation
    // walked element-by-element and emitted intermediate gaps after each
    // single-side advance. For samples with tied values *within* one side
    // (e.g. a=[1,1,2,2], b=[1,2]) the algorithm over-reported D — it saw
    // a 0.25 gap between the two one-side advances of a[0] and b[0], even
    // though the true CDF step function jumps at distinct x-breakpoints
    // only. Concretely, on a=[1,1,2,2], b=[1,2] the textbook D is 0; the
    // old code returned 0.25. This silently affected E9039 (numeric drift),
    // E9110 (vocab KS) and E9112 (language-shift KS), all of which feed
    // KS-D over heavily-tied frequency-counted distributions.
    //
    // The corrected algorithm advances both pointers past *all* equal
    // values at the next distinct breakpoint, then measures the gap once.
    // Matches the textbook two-sample KS step-function CDF construction.
    while i < n || j < m {
        // Pick the next distinct break-x = min of the two heads.
        let next_x = match (a.get(i), b.get(j)) {
            (Some(&xa), Some(&xb)) => {
                if xa.total_cmp(&xb).is_le() {
                    xa
                } else {
                    xb
                }
            }
            (Some(&xa), None) => xa,
            (None, Some(&xb)) => xb,
            (None, None) => break,
        };
        // Advance i past every a value bit-equal (total_cmp) to next_x.
        while i < n && a[i].total_cmp(&next_x).is_eq() {
            i += 1;
        }
        // Same for j on the b side.
        while j < m && b[j].total_cmp(&next_x).is_eq() {
            j += 1;
        }
        let f_a = i as f64 / n_f;
        let f_b = j as f64 / m_f;
        let gap = (f_a - f_b).abs();
        if gap > d_max {
            d_max = gap;
        }
    }
    Some(d_max)
}

/// Deterministic quantile via a sorted copy. Uses linear interpolation
/// between adjacent indices (Hyndman & Fan method 7, the same as
/// numpy's default `quantile`). NaN values are filtered.
///
/// `q` must be in `[0, 1]`. Returns `None` for empty or all-NaN input.
pub fn quantile_f64(values: &[f64], q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        return None;
    }
    let mut s: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    if s.is_empty() {
        return None;
    }
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n == 1 {
        return Some(s[0]);
    }
    let h = q * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let frac = h - lo as f64;
    Some(s[lo] + frac * (s[hi] - s[lo]))
}

/// Median absolute deviation around the median.
/// Returns `None` if input has fewer than 1 valid value.
pub fn median_absolute_deviation(values: &[f64]) -> Option<f64> {
    let med = quantile_f64(values, 0.5)?;
    let deviations: Vec<f64> = values
        .iter()
        .filter(|x| !x.is_nan())
        .map(|x| (x - med).abs())
        .collect();
    quantile_f64(&deviations, 0.5)
}

/// Per-value outlier scores using **two complementary methods**:
///
/// - `iqr_score`: how many IQRs the value sits outside `[Q1, Q3]`.
/// - `mod_z_score`: modified Z-score with median + MAD (`0.6745 * (x - med) / MAD`).
///
/// `None` if the inputs are too degenerate (constant column → IQR = 0, MAD = 0).
#[derive(Clone, Debug, PartialEq)]
pub struct OutlierBaselines {
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub iqr: f64,
    pub mad: f64,
    /// 0.6745 / MAD; the per-unit multiplier for modified Z.
    pub mod_z_inv: f64,
}

pub fn outlier_baselines(values: &[f64]) -> Option<OutlierBaselines> {
    let q1 = quantile_f64(values, 0.25)?;
    let median = quantile_f64(values, 0.5)?;
    let q3 = quantile_f64(values, 0.75)?;
    let iqr = q3 - q1;
    let mad = median_absolute_deviation(values)?;
    // Avoid div-by-zero in mod-Z. If MAD is 0 (constant column at the
    // median), the score is ill-defined; downstream code treats this
    // as "no outliers."
    let mod_z_inv = if mad > 0.0 { 0.6745 / mad } else { 0.0 };
    Some(OutlierBaselines {
        q1,
        median,
        q3,
        iqr,
        mad,
        mod_z_inv,
    })
}

/// Compute deterministic, equal-width histogram bin counts.
///
/// `n_bins >= 1`. Edges are `[min, max]` inclusive on the right edge of
/// the last bin. Returns `None` if `min == max` (a constant column has
/// no informative histogram) or `n_bins == 0`.
pub fn equal_width_histogram(values: &[f64], n_bins: usize) -> Option<Vec<u64>> {
    if n_bins == 0 {
        return None;
    }
    let summary = summarize_f64(values);
    let (min, max) = match (summary.min, summary.max) {
        (Some(a), Some(b)) if a < b => (a, b),
        _ => return None,
    };
    let width = (max - min) / n_bins as f64;
    let mut counts = vec![0u64; n_bins];
    for &v in values {
        if v.is_nan() {
            continue;
        }
        let mut idx = ((v - min) / width) as i64;
        if idx < 0 {
            idx = 0;
        }
        if idx >= n_bins as i64 {
            idx = n_bins as i64 - 1;
        }
        counts[idx as usize] += 1;
    }
    Some(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_handles_nan_as_missing() {
        let xs = [1.0, 2.0, f64::NAN, 4.0];
        let s = summarize_f64(&xs);
        assert_eq!(s.n_total, 4);
        assert_eq!(s.n_valid, 3);
        assert_eq!(s.n_missing, 1);
        assert!((s.mean.unwrap() - (7.0 / 3.0)).abs() < 1e-12);
        assert_eq!(s.min, Some(1.0));
        assert_eq!(s.max, Some(4.0));
    }

    #[test]
    fn summary_of_empty_is_well_defined() {
        let s = summarize_f64(&[]);
        assert_eq!(s.n_total, 0);
        assert_eq!(s.n_valid, 0);
        assert!(s.mean.is_none());
    }

    #[test]
    fn summary_of_all_nan_is_empty() {
        let s = summarize_f64(&[f64::NAN, f64::NAN]);
        assert_eq!(s.n_valid, 0);
        assert_eq!(s.n_missing, 2);
        assert!(s.mean.is_none());
    }

    #[test]
    fn variance_uses_population_denominator() {
        // mean = 2, deviations = [-1, 0, 1], variance = 2/3.
        let s = summarize_f64(&[1.0, 2.0, 3.0]);
        assert!((s.variance.unwrap() - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn summary_is_deterministic_across_runs() {
        let xs: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        let a = summarize_f64(&xs);
        let b = summarize_f64(&xs);
        assert_eq!(a, b);
    }

    #[test]
    fn correlation_of_identical_is_one() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = xs.clone();
        let r = pearson_correlation(&xs, &ys).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn correlation_of_anti_correlated_is_minus_one() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|v| -v).collect();
        let r = pearson_correlation(&xs, &ys).unwrap();
        assert!((r + 1.0).abs() < 1e-12);
    }

    #[test]
    fn correlation_of_constant_is_none() {
        let xs = vec![1.0, 1.0, 1.0];
        let ys = vec![5.0, 6.0, 7.0];
        assert!(pearson_correlation(&xs, &ys).is_none());
    }

    #[test]
    fn histogram_uniform_is_roughly_flat() {
        let xs: Vec<f64> = (0..1000).map(|i| i as f64 / 999.0).collect();
        let hist = equal_width_histogram(&xs, 10).unwrap();
        assert_eq!(hist.len(), 10);
        let total: u64 = hist.iter().sum();
        assert_eq!(total, 1000);
        // Last bin gets the inclusive right edge so it can be 1 over.
        for &c in &hist {
            assert!(c >= 90 && c <= 110, "bin count {} out of expected range", c);
        }
    }

    #[test]
    fn histogram_constant_column_is_none() {
        assert!(equal_width_histogram(&[3.0, 3.0, 3.0], 5).is_none());
    }

    #[test]
    fn ks_d_identical_samples_is_zero() {
        let v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let d = ks_d_statistic(&v, &v).unwrap();
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn ks_d_disjoint_supports_is_one() {
        let a: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b: Vec<f64> = (100..150).map(|i| i as f64).collect();
        let d = ks_d_statistic(&a, &b).unwrap();
        assert!((d - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ks_d_shifted_distribution_is_intermediate() {
        let a: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0).collect();
        let b: Vec<f64> = (0..1000).map(|i| (i as f64) / 999.0 + 0.5).collect();
        let d = ks_d_statistic(&a, &b).unwrap();
        // A 0.5 horizontal shift on uniform [0,1] yields D ≈ 0.5.
        assert!((d - 0.5).abs() < 0.01);
    }

    #[test]
    fn ks_d_skips_nan() {
        let a = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let d = ks_d_statistic(&a, &b).unwrap();
        assert!(d.abs() < 1e-12, "NaN filtered, identical distributions → D=0");
    }

    #[test]
    fn ks_d_short_samples_return_none() {
        assert!(ks_d_statistic(&[1.0], &[2.0, 3.0]).is_none());
        assert!(ks_d_statistic(&[], &[]).is_none());
    }

    #[test]
    fn ks_d_is_deterministic_across_runs() {
        let a: Vec<f64> = (0..500).map(|i| (i as f64).sin()).collect();
        let b: Vec<f64> = (0..500).map(|i| (i as f64).cos()).collect();
        let d1 = ks_d_statistic(&a, &b).unwrap();
        let d2 = ks_d_statistic(&a, &b).unwrap();
        assert_eq!(d1.to_bits(), d2.to_bits(), "KS D bits must match exactly");
    }
}
