//! Descriptive statistics — population & sample variance, sd, se, median,
//! quantile, IQR, skewness, kurtosis, z-score, standardize.
//!
//! # Determinism Contract
//!
//! - All reductions use `KahanAccumulatorF64` from `cjc_repro`.
//! - Sorting uses `f64::total_cmp` for deterministic NaN handling.
//! - No `HashMap`, no `par_iter`, no OS randomness.
//! - Same input order => bit-identical output.

use cjc_repro::KahanAccumulatorF64;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Kahan-stable mean of a slice.
fn kahan_mean(data: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for &x in data {
        acc.add(x);
    }
    acc.finalize() / data.len() as f64
}

/// Clone and sort using total_cmp for deterministic NaN ordering.
fn sorted_copy(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Population variance: sum((xi - mean)^2) / n.
/// Two-pass: first Kahan mean, then Kahan sum of squared deviations.
/// Variance (sample, N-1 denominator — R/pandas default).
/// For single element, returns 0.
pub fn variance(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Err("variance: empty data".into());
    }
    if data.len() == 1 {
        return Ok(0.0);
    }
    let mean = kahan_mean(data);
    let mut acc = KahanAccumulatorF64::new();
    for &x in data {
        let d = x - mean;
        acc.add(d * d);
    }
    Ok(acc.finalize() / (data.len() - 1) as f64)
}

/// Sample variance: alias for variance() (both use N-1 denominator).
pub fn sample_variance(data: &[f64]) -> Result<f64, String> {
    variance(data)
}

/// Population variance: sum((xi - mean)^2) / N.
pub fn pop_variance(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Err("pop_variance: empty data".into());
    }
    let mean = kahan_mean(data);
    let mut acc = KahanAccumulatorF64::new();
    for &x in data {
        let d = x - mean;
        acc.add(d * d);
    }
    Ok(acc.finalize() / data.len() as f64)
}

/// Standard deviation (sample, N-1 denominator — R/pandas default).
pub fn sd(data: &[f64]) -> Result<f64, String> {
    variance(data).map(|v| v.sqrt())
}

/// Sample standard deviation: alias for sd() (both use N-1 denominator).
pub fn sample_sd(data: &[f64]) -> Result<f64, String> {
    sd(data)
}

/// Population standard deviation: sqrt(pop_variance).
pub fn pop_sd(data: &[f64]) -> Result<f64, String> {
    pop_variance(data).map(|v| v.sqrt())
}

/// Standard error of the mean: sample_sd / sqrt(n).
pub fn se(data: &[f64]) -> Result<f64, String> {
    let s = sample_sd(data)?;
    Ok(s / (data.len() as f64).sqrt())
}

/// Median: middle value of sorted data.
/// For even n, average of two middle values.
/// Clones and sorts internally — never mutates input.
pub fn median(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Ok(f64::NAN);
    }
    let sorted = sorted_copy(data);
    let n = sorted.len();
    if n % 2 == 1 {
        Ok(sorted[n / 2])
    } else {
        Ok((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Quantile at probability p (0.0 to 1.0).
/// Linear interpolation between adjacent ranks (R type 7 / NumPy default).
pub fn quantile(data: &[f64], p: f64) -> Result<f64, String> {
    if data.is_empty() {
        return Err("quantile: empty data".into());
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(format!("quantile: p must be in [0, 1], got {p}"));
    }
    let sorted = sorted_copy(data);
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }
    // R type 7: index = (n-1)*p
    let idx = (n as f64 - 1.0) * p;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if lo == hi || hi >= n {
        Ok(sorted[lo])
    } else {
        Ok(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
    }
}

/// Interquartile range: Q3 - Q1.
pub fn iqr(data: &[f64]) -> Result<f64, String> {
    let q1 = quantile(data, 0.25)?;
    let q3 = quantile(data, 0.75)?;
    Ok(q3 - q1)
}

/// Skewness (Fisher's definition): E[(X-mu)^3] / sigma^3.
pub fn skewness(data: &[f64]) -> Result<f64, String> {
    if data.len() < 3 {
        return Err("skewness: need at least 3 elements".into());
    }
    let mean = kahan_mean(data);
    let n = data.len() as f64;
    let mut m2_acc = KahanAccumulatorF64::new();
    let mut m3_acc = KahanAccumulatorF64::new();
    for &x in data {
        let d = x - mean;
        m2_acc.add(d * d);
        m3_acc.add(d * d * d);
    }
    let m2 = m2_acc.finalize() / n;
    let m3 = m3_acc.finalize() / n;
    let sigma3 = m2.powf(1.5);
    if sigma3 == 0.0 {
        return Err("skewness: zero variance".into());
    }
    Ok(m3 / sigma3)
}

/// Kurtosis (excess kurtosis, Fisher's): E[(X-mu)^4] / sigma^4 - 3.
pub fn kurtosis(data: &[f64]) -> Result<f64, String> {
    if data.len() < 4 {
        return Err("kurtosis: need at least 4 elements".into());
    }
    let mean = kahan_mean(data);
    let n = data.len() as f64;
    let mut m2_acc = KahanAccumulatorF64::new();
    let mut m4_acc = KahanAccumulatorF64::new();
    for &x in data {
        let d = x - mean;
        let d2 = d * d;
        m2_acc.add(d2);
        m4_acc.add(d2 * d2);
    }
    let m2 = m2_acc.finalize() / n;
    let m4 = m4_acc.finalize() / n;
    if m2 == 0.0 {
        return Err("kurtosis: zero variance".into());
    }
    Ok(m4 / (m2 * m2) - 3.0)
}

/// Z-scores: (xi - mean) / sd for each element.
pub fn z_score(data: &[f64]) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Err("z_score: empty data".into());
    }
    let mean = kahan_mean(data);
    let s = sd(data)?;
    if s == 0.0 {
        return Err("z_score: zero standard deviation".into());
    }
    Ok(data.iter().map(|&x| (x - mean) / s).collect())
}

/// Min-max normalization: (xi - min) / (max - min).
pub fn standardize(data: &[f64]) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Err("standardize: empty data".into());
    }
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &x in data {
        if x < min_val { min_val = x; }
        if x > max_val { max_val = x; }
    }
    let range = max_val - min_val;
    if range == 0.0 {
        return Ok(vec![0.0; data.len()]);
    }
    Ok(data.iter().map(|&x| (x - min_val) / range).collect())
}

/// Number of distinct values in the data.
/// Uses sorted unique comparison for determinism (no HashMap).
pub fn n_distinct(data: &[f64]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let sorted = sorted_copy(data);
    let mut count = 1;
    for i in 1..sorted.len() {
        if sorted[i].to_bits() != sorted[i - 1].to_bits() {
            count += 1;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Correlation & Covariance (Sprint 2 additions)
// ---------------------------------------------------------------------------

/// Pearson correlation coefficient between two arrays.
/// cor(x, y) = cov(x,y) / (sd(x) * sd(y))
pub fn cor(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("cor: arrays must have same length".into());
    }
    if x.len() < 2 {
        return Err("cor: need at least 2 elements".into());
    }
    let mean_x = kahan_mean(x);
    let mean_y = kahan_mean(y);
    let mut cov_acc = KahanAccumulatorF64::new();
    let mut var_x_acc = KahanAccumulatorF64::new();
    let mut var_y_acc = KahanAccumulatorF64::new();
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov_acc.add(dx * dy);
        var_x_acc.add(dx * dx);
        var_y_acc.add(dy * dy);
    }
    let denom = (var_x_acc.finalize() * var_y_acc.finalize()).sqrt();
    if denom == 0.0 {
        return Err("cor: zero variance in one or both arrays".into());
    }
    Ok(cov_acc.finalize() / denom)
}

/// Population covariance: sum((xi-mx)(yi-my)) / n.
pub fn cov(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("cov: arrays must have same length".into());
    }
    if x.is_empty() {
        return Err("cov: empty data".into());
    }
    let mean_x = kahan_mean(x);
    let mean_y = kahan_mean(y);
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..x.len() {
        acc.add((x[i] - mean_x) * (y[i] - mean_y));
    }
    Ok(acc.finalize() / x.len() as f64)
}

/// Sample covariance: sum((xi-mx)(yi-my)) / (n-1).
pub fn sample_cov(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("sample_cov: arrays must have same length".into());
    }
    if x.len() < 2 {
        return Err("sample_cov: need at least 2 elements".into());
    }
    let mean_x = kahan_mean(x);
    let mean_y = kahan_mean(y);
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..x.len() {
        acc.add((x[i] - mean_x) * (y[i] - mean_y));
    }
    Ok(acc.finalize() / (x.len() - 1) as f64)
}

/// Correlation matrix for a set of variables (columns).
/// Returns flat Vec<f64> of n x n correlation matrix.
pub fn cor_matrix(vars: &[&[f64]]) -> Result<Vec<f64>, String> {
    let n = vars.len();
    if n == 0 {
        return Err("cor_matrix: no variables".into());
    }
    let len = vars[0].len();
    for v in vars {
        if v.len() != len {
            return Err("cor_matrix: all variables must have same length".into());
        }
    }
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        result[i * n + i] = 1.0; // diagonal
        for j in (i + 1)..n {
            let r = cor(vars[i], vars[j])?;
            result[i * n + j] = r;
            result[j * n + i] = r;
        }
    }
    Ok(result)
}

/// Covariance matrix.
/// Returns flat Vec<f64> of n x n covariance matrix.
pub fn cov_matrix(vars: &[&[f64]]) -> Result<Vec<f64>, String> {
    let n = vars.len();
    if n == 0 {
        return Err("cov_matrix: no variables".into());
    }
    let len = vars[0].len();
    for v in vars {
        if v.len() != len {
            return Err("cov_matrix: all variables must have same length".into());
        }
    }
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let c = cov(vars[i], vars[j])?;
            result[i * n + j] = c;
            result[j * n + i] = c;
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Cumulative operations & ranking (Sprint 5)
// ---------------------------------------------------------------------------

/// Cumulative sum with Kahan summation.
pub fn cumsum(data: &[f64]) -> Vec<f64> {
    let mut acc = KahanAccumulatorF64::new();
    let mut result = Vec::with_capacity(data.len());
    for &x in data {
        acc.add(x);
        result.push(acc.finalize());
    }
    result
}

/// Cumulative product.
pub fn cumprod(data: &[f64]) -> Vec<f64> {
    let mut prod = 1.0;
    let mut result = Vec::with_capacity(data.len());
    for &x in data {
        prod *= x;
        result.push(prod);
    }
    result
}

/// Cumulative max.
pub fn cummax(data: &[f64]) -> Vec<f64> {
    let mut mx = f64::NEG_INFINITY;
    let mut result = Vec::with_capacity(data.len());
    for &x in data {
        if x > mx { mx = x; }
        result.push(mx);
    }
    result
}

/// Cumulative min.
pub fn cummin(data: &[f64]) -> Vec<f64> {
    let mut mn = f64::INFINITY;
    let mut result = Vec::with_capacity(data.len());
    for &x in data {
        if x < mn { mn = x; }
        result.push(mn);
    }
    result
}

/// Lag: shift values forward by n positions, fill with NaN.
pub fn lag(data: &[f64], n: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        if i < n {
            result.push(f64::NAN);
        } else {
            result.push(data[i - n]);
        }
    }
    result
}

/// Lead: shift values backward by n positions, fill with NaN.
pub fn lead(data: &[f64], n: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        if i + n < data.len() {
            result.push(data[i + n]);
        } else {
            result.push(f64::NAN);
        }
    }
    result
}

/// Rank (average ties). Returns 1-based ranks.
/// DETERMINISM: uses stable sort with index tracking.
pub fn rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1.to_bits() == indexed[i].1.to_bits() {
            j += 1;
        }
        // Average rank for ties (1-based)
        let avg = (i + 1..=j).map(|r| r as f64).sum::<f64>() / (j - i) as f64;
        for k in i..j {
            ranks[indexed[k].0] = avg;
        }
        i = j;
    }
    ranks
}

/// Dense rank (no gaps for ties). Returns 1-based ranks.
pub fn dense_rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut ranks = vec![0.0; n];
    let mut current_rank = 1.0;
    for i in 0..n {
        if i > 0 && indexed[i].1.to_bits() != indexed[i - 1].1.to_bits() {
            current_rank += 1.0;
        }
        ranks[indexed[i].0] = current_rank;
    }
    ranks
}

/// Row number (sequential, tie-broken by original position — stable).
pub fn row_number(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut ranks = vec![0.0; n];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        ranks[orig_idx] = (rank + 1) as f64;
    }
    ranks
}

/// Histogram: bin data into n equal-width bins.
/// Returns (bin_edges: Vec<f64>, counts: Vec<usize>).
pub fn histogram(data: &[f64], n_bins: usize) -> Result<(Vec<f64>, Vec<usize>), String> {
    if data.is_empty() {
        return Err("histogram: empty data".into());
    }
    if n_bins == 0 {
        return Err("histogram: n_bins must be > 0".into());
    }
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &x in data {
        if x < min_val { min_val = x; }
        if x > max_val { max_val = x; }
    }
    let range = max_val - min_val;
    let width = if range == 0.0 { 1.0 } else { range / n_bins as f64 };
    let mut edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        edges.push(min_val + width * i as f64);
    }
    let mut counts = vec![0usize; n_bins];
    for &x in data {
        let bin = if range == 0.0 {
            0
        } else {
            let b = ((x - min_val) / width).floor() as usize;
            b.min(n_bins - 1) // last edge inclusive
        };
        counts[bin] += 1;
    }
    Ok((edges, counts))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_basic() {
        // variance uses sample (N-1) denominator
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = variance(&data).unwrap();
        // mean=5, sum_sq_dev=32, sample_var=32/7≈4.571
        assert!((v - 32.0 / 7.0).abs() < 1e-12, "expected 32/7, got {v}");
    }

    #[test]
    fn test_sd_basic() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = sd(&data).unwrap();
        let expected = (32.0_f64 / 7.0).sqrt();
        assert!((s - expected).abs() < 1e-12, "expected sqrt(32/7), got {s}");
    }

    #[test]
    fn test_median_odd() {
        let data = [1.0, 3.0, 5.0];
        assert_eq!(median(&data).unwrap(), 3.0);
    }

    #[test]
    fn test_median_even() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&data).unwrap(), 2.5);
    }

    #[test]
    fn test_quantile_basic() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let q25 = quantile(&data, 0.25).unwrap();
        let q50 = quantile(&data, 0.5).unwrap();
        let q75 = quantile(&data, 0.75).unwrap();
        assert!((q50 - 50.5).abs() < 1e-12);
        assert!((q25 - 25.75).abs() < 1e-12);
        assert!((q75 - 75.25).abs() < 1e-12);
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric data around 0 → skewness ≈ 0
        let data = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let s = skewness(&data).unwrap();
        assert!(s.abs() < 1e-12, "skewness should be ~0, got {s}");
    }

    #[test]
    fn test_kurtosis_normal() {
        // Uniform[-1,1] has kurtosis -6/5 = -1.2
        let data: Vec<f64> = (0..1000).map(|i| -1.0 + 2.0 * (i as f64 / 999.0)).collect();
        let k = kurtosis(&data).unwrap();
        assert!((k - (-1.2)).abs() < 0.05, "excess kurtosis got {k}");
    }

    #[test]
    fn test_z_score_basic() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let z = z_score(&data).unwrap();
        // Mean of z-scores should be ~0
        let z_mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert!(z_mean.abs() < 1e-12);
    }

    #[test]
    fn test_determinism() {
        let data = [1.1, 2.2, 3.3, 4.4, 5.5];
        let v1 = variance(&data).unwrap();
        let v2 = variance(&data).unwrap();
        assert_eq!(v1.to_bits(), v2.to_bits());
    }

    #[test]
    fn test_empty_data_error() {
        assert!(variance(&[]).is_err());
        assert!(sd(&[]).is_err());
        // median returns NaN for empty data (not error)
        assert!(median(&[]).unwrap().is_nan());
        assert!(quantile(&[], 0.5).is_err());
    }

    #[test]
    fn test_kahan_stability() {
        let data: Vec<f64> = (0..10000).map(|_| 0.1).collect();
        let v = variance(&data).unwrap();
        // All identical values → variance should be exactly 0
        assert!(v.abs() < 1e-20, "variance of identical values should be ~0, got {v}");
    }

    #[test]
    fn test_cor_perfect() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = cor(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cor_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        let r = cor(&x, &y).unwrap();
        assert!((r - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_cov_basic() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let c = cov(&x, &y).unwrap();
        // cov = mean((x-mx)(y-my)) = ((-1)(-1) + 0*0 + 1*1)/3 = 2/3
        assert!((c - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(cumsum(&data), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumprod() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(cumprod(&data), vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_rank_basic() {
        let data = [3.0, 1.0, 4.0, 1.0];
        let r = rank(&data);
        // 1.0 appears twice at positions 1,3 → avg rank = 1.5
        // 3.0 → rank 3
        // 4.0 → rank 4
        assert_eq!(r, vec![3.0, 1.5, 4.0, 1.5]);
    }

    #[test]
    fn test_dense_rank_basic() {
        let data = [3.0, 1.0, 4.0, 1.0];
        let r = dense_rank(&data);
        assert_eq!(r, vec![2.0, 1.0, 3.0, 1.0]);
    }

    #[test]
    fn test_histogram_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (edges, counts) = histogram(&data, 5).unwrap();
        assert_eq!(edges.len(), 6);
        let total: usize = counts.iter().sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_lag_lead() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let lagged = lag(&data, 2);
        assert!(lagged[0].is_nan());
        assert!(lagged[1].is_nan());
        assert_eq!(lagged[2], 1.0);

        let led = lead(&data, 2);
        assert_eq!(led[0], 3.0);
        assert!(led[3].is_nan());
        assert!(led[4].is_nan());
    }

    #[test]
    fn test_standardize() {
        let data = [0.0, 5.0, 10.0];
        let s = standardize(&data).unwrap();
        assert_eq!(s, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_n_distinct() {
        let data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        assert_eq!(n_distinct(&data), 3);
    }
}
