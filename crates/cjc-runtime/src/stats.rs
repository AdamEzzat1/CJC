//! Descriptive statistics — population & sample variance, sd, se, median,
//! quantile, IQR, skewness, kurtosis, z-score, standardize.
//!
//! # Determinism Contract
//!
//! - Unordered reductions use `BinnedAccumulatorF64` for order-invariant,
//!   bit-identical results regardless of input order.
//! - Cumulative (ordered) operations use `KahanAccumulatorF64` where the
//!   addition order IS the semantics.
//! - Sorting uses `f64::total_cmp` for deterministic NaN handling.
//! - No `HashMap`, no `par_iter`, no OS randomness.
//! - Same input => bit-identical output.

use crate::accumulator::BinnedAccumulatorF64;
use cjc_repro::KahanAccumulatorF64;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Order-invariant mean of a slice using [`BinnedAccumulatorF64`].
fn binned_mean(data: &[f64]) -> f64 {
    let mut acc = BinnedAccumulatorF64::new();
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

/// Variance (sample, N-1 denominator — R/pandas default).
/// Two-pass: first binned mean, then binned sum of squared deviations.
/// For single element, returns 0.
pub fn variance(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Err("variance: empty data".into());
    }
    if data.len() == 1 {
        return Ok(0.0);
    }
    let mean = binned_mean(data);
    let mut acc = BinnedAccumulatorF64::new();
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
    let mean = binned_mean(data);
    let mut acc = BinnedAccumulatorF64::new();
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
    let mean = binned_mean(data);
    let n = data.len() as f64;
    let mut m2_acc = BinnedAccumulatorF64::new();
    let mut m3_acc = BinnedAccumulatorF64::new();
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
    let mean = binned_mean(data);
    let n = data.len() as f64;
    let mut m2_acc = BinnedAccumulatorF64::new();
    let mut m4_acc = BinnedAccumulatorF64::new();
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
    let mean = binned_mean(data);
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
    let mean_x = binned_mean(x);
    let mean_y = binned_mean(y);
    let mut cov_acc = BinnedAccumulatorF64::new();
    let mut var_x_acc = BinnedAccumulatorF64::new();
    let mut var_y_acc = BinnedAccumulatorF64::new();
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
    let mean_x = binned_mean(x);
    let mean_y = binned_mean(y);
    let mut acc = BinnedAccumulatorF64::new();
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
    let mean_x = binned_mean(x);
    let mean_y = binned_mean(y);
    let mut acc = BinnedAccumulatorF64::new();
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
// Weighted & robust statistics (Phase B, sub-sprint B1)
// ---------------------------------------------------------------------------

/// Weighted mean: sum(data[i] * weights[i]) / sum(weights).
/// Uses binned accumulation for both numerator and denominator.
pub fn weighted_mean(data: &[f64], weights: &[f64]) -> Result<f64, String> {
    if data.len() != weights.len() {
        return Err("weighted_mean: data and weights must have same length".into());
    }
    if data.is_empty() {
        return Err("weighted_mean: empty data".into());
    }
    let mut num_acc = BinnedAccumulatorF64::new();
    let mut den_acc = BinnedAccumulatorF64::new();
    for i in 0..data.len() {
        num_acc.add(data[i] * weights[i]);
        den_acc.add(weights[i]);
    }
    let denom = den_acc.finalize();
    if denom == 0.0 {
        return Err("weighted_mean: weights sum to zero".into());
    }
    Ok(num_acc.finalize() / denom)
}

/// Weighted variance: sum(w[i] * (x[i] - weighted_mean)^2) / sum(w).
/// Two-pass: first weighted mean (binned), then binned sum of squared deviations.
pub fn weighted_var(data: &[f64], weights: &[f64]) -> Result<f64, String> {
    if data.len() != weights.len() {
        return Err("weighted_var: data and weights must have same length".into());
    }
    if data.is_empty() {
        return Err("weighted_var: empty data".into());
    }
    let wm = weighted_mean(data, weights)?;
    let mut num_acc = BinnedAccumulatorF64::new();
    let mut den_acc = BinnedAccumulatorF64::new();
    for i in 0..data.len() {
        let d = data[i] - wm;
        num_acc.add(weights[i] * d * d);
        den_acc.add(weights[i]);
    }
    let denom = den_acc.finalize();
    if denom == 0.0 {
        return Err("weighted_var: weights sum to zero".into());
    }
    Ok(num_acc.finalize() / denom)
}

/// Trimmed mean: mean of data with `proportion` fraction removed from each tail.
/// proportion=0.1 removes bottom 10% and top 10%, computing mean of middle 80%.
pub fn trimmed_mean(data: &[f64], proportion: f64) -> Result<f64, String> {
    if data.is_empty() {
        return Err("trimmed_mean: empty data".into());
    }
    if proportion < 0.0 || proportion >= 0.5 {
        return Err("trimmed_mean: proportion must be in [0, 0.5)".into());
    }
    let sorted = sorted_copy(data);
    let n = sorted.len();
    let trim = (n as f64 * proportion).floor() as usize;
    let trimmed = &sorted[trim..n - trim];
    if trimmed.is_empty() {
        return Err("trimmed_mean: all data trimmed".into());
    }
    Ok(binned_mean(trimmed))
}

/// Winsorize: replace values below the `proportion` quantile with the lower
/// boundary, and values above the `(1-proportion)` quantile with the upper boundary.
pub fn winsorize(data: &[f64], proportion: f64) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Err("winsorize: empty data".into());
    }
    if proportion < 0.0 || proportion >= 0.5 {
        return Err("winsorize: proportion must be in [0, 0.5)".into());
    }
    if proportion == 0.0 {
        return Ok(data.to_vec());
    }
    let lo = quantile(data, proportion)?;
    let hi = quantile(data, 1.0 - proportion)?;
    Ok(data.iter().map(|&x| {
        if x < lo { lo } else if x > hi { hi } else { x }
    }).collect())
}

/// Median absolute deviation: median(|x[i] - median(x)|).
/// Does NOT multiply by 1.4826 scaling factor.
pub fn mad(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Err("mad: empty data".into());
    }
    let med = median(data)?;
    let abs_devs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    median(&abs_devs)
}

/// Mode: most frequent value. Ties broken by smallest value.
/// Uses bit-exact comparison via to_bits() on a sorted copy.
pub fn mode(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Err("mode: empty data".into());
    }
    let sorted = sorted_copy(data);
    let mut best_val = sorted[0];
    let mut best_count = 1usize;
    let mut cur_val = sorted[0];
    let mut cur_count = 1usize;
    for i in 1..sorted.len() {
        if sorted[i].to_bits() == cur_val.to_bits() {
            cur_count += 1;
        } else {
            if cur_count > best_count {
                best_count = cur_count;
                best_val = cur_val;
            }
            cur_val = sorted[i];
            cur_count = 1;
        }
    }
    if cur_count > best_count {
        best_val = cur_val;
    }
    Ok(best_val)
}

/// Percentile rank: fraction of data values strictly less than the given value,
/// plus half the fraction equal to the value. Returns a value in [0, 1].
pub fn percentile_rank(data: &[f64], value: f64) -> Result<f64, String> {
    if data.is_empty() {
        return Err("percentile_rank: empty data".into());
    }
    let n = data.len() as f64;
    let mut below = 0usize;
    let mut equal = 0usize;
    for &x in data {
        match x.total_cmp(&value) {
            std::cmp::Ordering::Less => below += 1,
            std::cmp::Ordering::Equal => equal += 1,
            std::cmp::Ordering::Greater => {}
        }
    }
    Ok((below as f64 + 0.5 * equal as f64) / n)
}

// ---------------------------------------------------------------------------
// Rank correlations & partial correlation (Phase B, sub-sprint B2)
// ---------------------------------------------------------------------------

/// Spearman rank correlation: Pearson correlation of the ranks of x and y.
pub fn spearman_cor(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("spearman_cor: arrays must have same length".into());
    }
    if x.len() < 2 {
        return Err("spearman_cor: need at least 2 elements".into());
    }
    let rx = rank(x);
    let ry = rank(y);
    cor(&rx, &ry)
}

/// Kendall tau-b correlation coefficient with tie adjustment.
/// O(n^2) pairwise comparison for determinism.
pub fn kendall_cor(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("kendall_cor: arrays must have same length".into());
    }
    let n = x.len();
    if n < 2 {
        return Err("kendall_cor: need at least 2 elements".into());
    }
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut ties_x: i64 = 0;
    let mut ties_y: i64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i].total_cmp(&x[j]);
            let dy = y[i].total_cmp(&y[j]);
            match (dx, dy) {
                (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {
                    ties_x += 1;
                    ties_y += 1;
                }
                (std::cmp::Ordering::Equal, _) => { ties_x += 1; }
                (_, std::cmp::Ordering::Equal) => { ties_y += 1; }
                _ => {
                    if dx == dy { concordant += 1; } else { discordant += 1; }
                }
            }
        }
    }
    let n0 = (n * (n - 1) / 2) as f64;
    let denom = ((n0 - ties_x as f64) * (n0 - ties_y as f64)).sqrt();
    if denom == 0.0 {
        return Err("kendall_cor: no variation in one or both arrays".into());
    }
    Ok((concordant - discordant) as f64 / denom)
}

/// Partial correlation: correlation of x and y controlling for z.
pub fn partial_cor(x: &[f64], y: &[f64], z: &[f64]) -> Result<f64, String> {
    let rxy = cor(x, y)?;
    let rxz = cor(x, z)?;
    let ryz = cor(y, z)?;
    let denom_x = (1.0 - rxz * rxz).sqrt();
    let denom_y = (1.0 - ryz * ryz).sqrt();
    if denom_x == 0.0 || denom_y == 0.0 {
        return Err("partial_cor: perfect correlation with control variable".into());
    }
    Ok((rxy - rxz * ryz) / (denom_x * denom_y))
}

/// Confidence interval for Pearson correlation using Fisher z-transform.
/// Returns (lower_bound, upper_bound).
pub fn cor_ci(x: &[f64], y: &[f64], alpha: f64) -> Result<(f64, f64), String> {
    if x.len() != y.len() {
        return Err("cor_ci: arrays must have same length".into());
    }
    let n = x.len();
    if n < 4 {
        return Err("cor_ci: need at least 4 elements".into());
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err("cor_ci: alpha must be in (0, 1)".into());
    }
    let r = cor(x, y)?;
    let z_r = r.atanh(); // Fisher z-transform
    let se = 1.0 / ((n as f64 - 3.0).sqrt());
    let z_crit = crate::distributions::normal_ppf(1.0 - alpha / 2.0)?;
    let lo = (z_r - z_crit * se).tanh();
    let hi = (z_r + z_crit * se).tanh();
    Ok((lo, hi))
}

// ---------------------------------------------------------------------------
// Analyst QoL extensions (Phase B, sub-sprint B5)
// ---------------------------------------------------------------------------

/// Divide data into n roughly equal groups (ntile/quantile binning).
/// Returns 1-based group assignments matching original data order.
pub fn ntile(data: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Err("ntile: empty data".into());
    }
    if n == 0 {
        return Err("ntile: n must be > 0".into());
    }
    let len = data.len();
    // Sort by value with stable index tie-breaking
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut result = vec![0.0; len];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        let group = (rank * n / len) + 1;
        result[orig_idx] = group as f64;
    }
    Ok(result)
}

/// Percent rank: (rank - 1) / (n - 1), range [0, 1].
/// Uses average-tie ranking from existing rank() function.
pub fn percent_rank_fn(data: &[f64]) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Err("percent_rank: empty data".into());
    }
    let n = data.len();
    if n == 1 {
        return Ok(vec![0.0]);
    }
    let ranks = rank(data);
    Ok(ranks.iter().map(|&r| (r - 1.0) / (n as f64 - 1.0)).collect())
}

/// Cumulative distribution: count(x_i <= x_j) / n for each x_j.
pub fn cume_dist(data: &[f64]) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Err("cume_dist: empty data".into());
    }
    let n = data.len();
    // Sort with indices
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut result = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1.to_bits() == indexed[i].1.to_bits() {
            j += 1;
        }
        let cd = j as f64 / n as f64;
        for k in i..j {
            result[indexed[k].0] = cd;
        }
        i = j;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Selection primitives (Bastion ABI)
// ---------------------------------------------------------------------------

/// Introselect: partition-based O(n) expected selection of the k-th smallest
/// element. Operates on a mutable slice and partially reorders it so that
/// `data[k]` holds the k-th smallest value (0-indexed), all elements
/// `data[..k]` are <= data[k], and all `data[k+1..]` are >= data[k].
///
/// Uses Rust's `select_nth_unstable_by` with `total_cmp` for deterministic
/// NaN handling (NaN sorts as greater-than everything).
///
/// # Determinism Contract
/// Same input slice + same k => identical partial reorder and return value.
/// NaN-safe via total_cmp.
///
/// # Panics
/// Panics if k >= data.len().
pub fn nth_element(data: &mut [f64], k: usize) -> f64 {
    data.select_nth_unstable_by(k, f64::total_cmp);
    data[k]
}

/// Non-mutating nth_element: clones data, selects k-th element, returns it.
/// O(n) expected time, O(n) space for the clone.
pub fn nth_element_copy(data: &[f64], k: usize) -> Result<f64, String> {
    if data.is_empty() {
        return Err("nth_element: empty data".into());
    }
    if k >= data.len() {
        return Err(format!("nth_element: k={k} out of bounds (n={})", data.len()));
    }
    let mut buf = data.to_vec();
    Ok(nth_element(&mut buf, k))
}

/// O(n) median using introselect instead of O(n log n) sort.
/// For even n, selects both middle elements via two partial sorts.
pub fn median_fast(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Ok(f64::NAN);
    }
    let n = data.len();
    let mut buf = data.to_vec();
    if n % 2 == 1 {
        Ok(nth_element(&mut buf, n / 2))
    } else {
        // For even n, we need elements at n/2-1 and n/2.
        // After nth_element(n/2-1), buf[..n/2-1] are all <= buf[n/2-1],
        // so the element at n/2 is the min of buf[n/2..].
        let lo = nth_element(&mut buf, n / 2 - 1);
        // buf[n/2..] are all >= lo; find min of that subslice
        let hi = buf[n / 2..].iter().copied().fold(f64::INFINITY, f64::min);
        Ok((lo + hi) / 2.0)
    }
}

/// O(n) quantile using introselect (R type 7 interpolation).
pub fn quantile_fast(data: &[f64], p: f64) -> Result<f64, String> {
    if data.is_empty() {
        return Err("quantile_fast: empty data".into());
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(format!("quantile_fast: p must be in [0, 1], got {p}"));
    }
    let n = data.len();
    if n == 1 {
        return Ok(data[0]);
    }
    let idx = (n as f64 - 1.0) * p;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;

    let mut buf = data.to_vec();
    let lo_val = nth_element(&mut buf, lo);
    if lo == hi || hi >= n {
        Ok(lo_val)
    } else {
        // After nth_element(lo), buf[lo+1..] are >= buf[lo].
        // Find min of buf[lo+1..] which is the (lo+1)-th element.
        let hi_val = buf[lo + 1..].iter().copied().fold(f64::INFINITY, f64::min);
        Ok(lo_val * (1.0 - frac) + hi_val * frac)
    }
}

// ---------------------------------------------------------------------------
// Sampling primitives (Bastion ABI)
// ---------------------------------------------------------------------------

/// Generate k random indices in [0, n) with or without replacement.
///
/// Uses CJC's deterministic SplitMix64 RNG via seed.
///
/// # Determinism Contract
/// Same (n, k, replace, seed) => identical index vector.
///
/// # Panics / Errors
/// - If !replace && k > n, returns error.
/// - If n == 0 or k == 0, returns empty vec.
pub fn sample_indices(n: usize, k: usize, replace: bool, seed: u64) -> Result<Vec<usize>, String> {
    if n == 0 || k == 0 {
        return Ok(Vec::new());
    }
    let mut rng = cjc_repro::Rng::seeded(seed);

    if replace {
        // With replacement: just sample k indices
        let mut result = Vec::with_capacity(k);
        for _ in 0..k {
            let idx = (rng.next_f64() * n as f64) as usize;
            result.push(idx.min(n - 1)); // clamp for edge case
        }
        Ok(result)
    } else {
        if k > n {
            return Err(format!(
                "sample_indices: k={k} > n={n} without replacement"
            ));
        }
        // Fisher-Yates partial shuffle for k <= n
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + ((rng.next_f64() * (n - i) as f64) as usize).min(n - i - 1);
            pool.swap(i, j);
        }
        pool.truncate(k);
        Ok(pool)
    }
}

/// Boolean mask selection: return elements of data where mask is true.
///
/// # Determinism Contract
/// Same input => identical output. Order preserved.
pub fn filter_mask(data: &[f64], mask: &[bool]) -> Result<Vec<f64>, String> {
    if data.len() != mask.len() {
        return Err(format!(
            "filter_mask: data len ({}) != mask len ({})",
            data.len(), mask.len()
        ));
    }
    Ok(data.iter().zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&v, _)| v)
        .collect())
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

    // --- B1: Weighted & Robust Statistics tests ---

    #[test]
    fn test_weighted_mean_uniform() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = [1.0, 1.0, 1.0, 1.0, 1.0];
        let wm = weighted_mean(&data, &weights).unwrap();
        assert!((wm - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_mean_skewed() {
        let data = [1.0, 2.0, 3.0];
        let weights = [3.0, 0.0, 0.0];
        let wm = weighted_mean(&data, &weights).unwrap();
        assert!((wm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_mean_empty() {
        assert!(weighted_mean(&[], &[]).is_err());
    }

    #[test]
    fn test_weighted_var_uniform() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let weights = vec![1.0; data.len()];
        let wv = weighted_var(&data, &weights).unwrap();
        let pv = pop_variance(&data).unwrap();
        assert!((wv - pv).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_10pct() {
        // 10 elements, trim 10% = 1 from each end
        let data = [100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -50.0];
        let tm = trimmed_mean(&data, 0.1).unwrap();
        // After sorting: [-50, 2, 3, 4, 5, 6, 7, 8, 9, 100], trim 1 each → [2,3,4,5,6,7,8,9]
        let expected = binned_mean(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!((tm - expected).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_zero() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let tm = trimmed_mean(&data, 0.0).unwrap();
        assert!((tm - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_invalid_proportion() {
        assert!(trimmed_mean(&[1.0, 2.0], 0.5).is_err());
        assert!(trimmed_mean(&[1.0, 2.0], -0.1).is_err());
    }

    #[test]
    fn test_winsorize_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let w = winsorize(&data, 0.1).unwrap();
        // 10% quantile ≈ 1.9, 90% quantile ≈ 9.1
        // Values at extremes should be clipped
        assert!(w.iter().all(|&x| x >= 1.0 && x <= 10.0));
    }

    #[test]
    fn test_winsorize_no_change() {
        let data = [1.0, 2.0, 3.0];
        let w = winsorize(&data, 0.0).unwrap();
        assert_eq!(w, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mad_symmetric() {
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let m = mad(&data).unwrap();
        // median = 0, deviations = [2,1,0,1,2], median of deviations = 1
        assert!((m - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mad_constant() {
        let data = [5.0, 5.0, 5.0];
        let m = mad(&data).unwrap();
        assert!((m - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_mode_simple() {
        let data = [1.0, 2.0, 2.0, 3.0];
        assert!((mode(&data).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_mode_tie() {
        // 1.0 and 2.0 each appear twice; smallest wins
        let data = [2.0, 1.0, 2.0, 1.0];
        assert!((mode(&data).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mode_single() {
        assert!((mode(&[42.0]).unwrap() - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_rank_median() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let pr = percentile_rank(&data, 3.0).unwrap();
        // 2 below, 1 equal: (2 + 0.5*1)/5 = 0.5
        assert!((pr - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_rank_min() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let pr = percentile_rank(&data, 1.0).unwrap();
        // 0 below, 1 equal: (0 + 0.5*1)/5 = 0.1
        assert!((pr - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_b1_determinism() {
        let data = [1.1, 2.2, 3.3, 4.4, 5.5];
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0];
        let wm1 = weighted_mean(&data, &weights).unwrap();
        let wm2 = weighted_mean(&data, &weights).unwrap();
        assert_eq!(wm1.to_bits(), wm2.to_bits());
        let m1 = mad(&data).unwrap();
        let m2 = mad(&data).unwrap();
        assert_eq!(m1.to_bits(), m2.to_bits());
    }

    // --- B2: Rank correlations & partial correlation tests ---

    #[test]
    fn test_spearman_perfect_monotone() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 20.0, 30.0, 40.0, 50.0];
        let r = spearman_cor(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_spearman_perfect_reverse() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [50.0, 40.0, 30.0, 20.0, 10.0];
        let r = spearman_cor(&x, &y).unwrap();
        assert!((r - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_spearman_nonlinear() {
        // x vs x^2: monotonic but nonlinear
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v * v).collect();
        let r = spearman_cor(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-12); // strictly monotone → rho = 1
    }

    #[test]
    fn test_spearman_equals_pearson_for_linear() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let sp = spearman_cor(&x, &y).unwrap();
        let pe = cor(&x, &y).unwrap();
        assert!((sp - pe).abs() < 1e-12);
    }

    #[test]
    fn test_kendall_concordant() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let t = kendall_cor(&x, &y).unwrap();
        assert!((t - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_kendall_discordant() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let t = kendall_cor(&x, &y).unwrap();
        assert!((t - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_kendall_with_ties() {
        let x = [1.0, 2.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let t = kendall_cor(&x, &y).unwrap();
        // n0=6, concordant=5, discordant=0, ties_x=1, ties_y=0
        // tau_b = 5 / sqrt(5 * 6) ≈ 0.9129
        assert!(t > 0.9 && t < 1.0);
    }

    #[test]
    fn test_kendall_known_values() {
        // Known example: x=[12,2,1,12,2], y=[1,4,7,1,0]
        let x = [12.0, 2.0, 1.0, 12.0, 2.0];
        let y = [1.0, 4.0, 7.0, 1.0, 0.0];
        let t = kendall_cor(&x, &y).unwrap();
        // Should be negative (generally inverse relationship)
        assert!(t < 0.0);
    }

    #[test]
    fn test_partial_cor_no_confounding() {
        // z independent of both x and y → partial_cor ≈ cor(x,y)
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let z = [5.0, 3.0, 1.0, 4.0, 2.0]; // shuffled, not correlated
        let pc = partial_cor(&x, &y, &z).unwrap();
        // cor(x,y) = 1.0, so partial_cor should be close to 1.0
        assert!(pc > 0.95);
    }

    #[test]
    fn test_cor_ci_contains_r() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let r = cor(&x, &y).unwrap();
        let (lo, hi) = cor_ci(&x, &y, 0.05).unwrap();
        assert!(lo <= r && r <= hi, "r={r} should be in [{lo}, {hi}]");
    }

    #[test]
    fn test_cor_ci_95_narrower_than_99() {
        // Use imperfect correlation to avoid atanh(1.0) = inf
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().enumerate().map(|(i, &v)| {
            v * 2.0 + 1.0 + if i % 3 == 0 { 0.5 } else { -0.3 }
        }).collect();
        let (lo95, hi95) = cor_ci(&x, &y, 0.05).unwrap();
        let (lo99, hi99) = cor_ci(&x, &y, 0.01).unwrap();
        assert!(hi95 - lo95 < hi99 - lo99);
    }

    #[test]
    fn test_b2_determinism() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 3.0, 4.0, 2.0, 1.0];
        let s1 = spearman_cor(&x, &y).unwrap();
        let s2 = spearman_cor(&x, &y).unwrap();
        assert_eq!(s1.to_bits(), s2.to_bits());
        let k1 = kendall_cor(&x, &y).unwrap();
        let k2 = kendall_cor(&x, &y).unwrap();
        assert_eq!(k1.to_bits(), k2.to_bits());
    }

    // --- B5: Analyst QoL extensions tests ---

    #[test]
    fn test_ntile_even() {
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let groups = ntile(&data, 4).unwrap();
        assert_eq!(groups, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_ntile_uneven() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let groups = ntile(&data, 3).unwrap();
        // 10 items, 3 groups: groups should be 1..=3, each element assigned
        assert!(groups.iter().all(|&g| g >= 1.0 && g <= 3.0));
        // Should have roughly equal counts
        let c1 = groups.iter().filter(|&&g| g == 1.0).count();
        let c2 = groups.iter().filter(|&&g| g == 2.0).count();
        let c3 = groups.iter().filter(|&&g| g == 3.0).count();
        assert!(c1 >= 3 && c1 <= 4);
        assert!(c2 >= 3 && c2 <= 4);
        assert!(c3 >= 3 && c3 <= 4);
    }

    #[test]
    fn test_ntile_n_equals_len() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let groups = ntile(&data, 5).unwrap();
        assert_eq!(groups, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_ntile_error_empty() {
        assert!(ntile(&[], 3).is_err());
    }

    #[test]
    fn test_ntile_error_zero() {
        assert!(ntile(&[1.0], 0).is_err());
    }

    #[test]
    fn test_percent_rank_sorted() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let pr = percent_rank_fn(&data).unwrap();
        assert!((pr[0] - 0.0).abs() < 1e-12);
        assert!((pr[1] - 0.25).abs() < 1e-12);
        assert!((pr[2] - 0.5).abs() < 1e-12);
        assert!((pr[3] - 0.75).abs() < 1e-12);
        assert!((pr[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_percent_rank_ties() {
        let data = [1.0, 2.0, 2.0, 4.0];
        let pr = percent_rank_fn(&data).unwrap();
        // ranks: [1.0, 2.5, 2.5, 4.0]
        // percent_rank = (rank - 1) / (n - 1) = (rank - 1) / 3
        assert!((pr[0] - 0.0).abs() < 1e-12);
        assert!((pr[1] - pr[2]).abs() < 1e-12); // tied values same percent rank
        assert!((pr[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_percent_rank_single() {
        let pr = percent_rank_fn(&[42.0]).unwrap();
        assert_eq!(pr, vec![0.0]);
    }

    #[test]
    fn test_cume_dist_sorted() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cd = cume_dist(&data).unwrap();
        assert!((cd[0] - 0.2).abs() < 1e-12);
        assert!((cd[1] - 0.4).abs() < 1e-12);
        assert!((cd[2] - 0.6).abs() < 1e-12);
        assert!((cd[3] - 0.8).abs() < 1e-12);
        assert!((cd[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cume_dist_ties() {
        let data = [1.0, 2.0, 2.0, 4.0];
        let cd = cume_dist(&data).unwrap();
        assert!((cd[0] - 0.25).abs() < 1e-12);
        // Both 2.0s should have same cume_dist = 3/4 = 0.75
        assert!((cd[1] - 0.75).abs() < 1e-12);
        assert!((cd[2] - 0.75).abs() < 1e-12);
        assert!((cd[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_b5_determinism() {
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let n1 = ntile(&data, 4).unwrap();
        let n2 = ntile(&data, 4).unwrap();
        for (a, b) in n1.iter().zip(n2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        let pr1 = percent_rank_fn(&data).unwrap();
        let pr2 = percent_rank_fn(&data).unwrap();
        for (a, b) in pr1.iter().zip(pr2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // -----------------------------------------------------------------------
    // Bastion ABI primitive tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nth_element_basic() {
        let mut data = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0];
        // k=0 => smallest
        assert_eq!(nth_element(&mut data.clone(), 0), 1.0);
        // k=8 => largest
        assert_eq!(nth_element(&mut data.clone(), 8), 9.0);
        // k=4 => median of 9 elements
        assert_eq!(nth_element(&mut data.clone(), 4), 5.0);
    }

    #[test]
    fn test_nth_element_copy_basic() {
        let data = [5.0, 2.0, 8.0, 1.0, 9.0];
        assert_eq!(nth_element_copy(&data, 0).unwrap(), 1.0);
        assert_eq!(nth_element_copy(&data, 2).unwrap(), 5.0);
        assert_eq!(nth_element_copy(&data, 4).unwrap(), 9.0);
    }

    #[test]
    fn test_nth_element_copy_errors() {
        assert!(nth_element_copy(&[], 0).is_err());
        assert!(nth_element_copy(&[1.0], 1).is_err());
    }

    #[test]
    fn test_nth_element_single() {
        let mut data = vec![42.0];
        assert_eq!(nth_element(&mut data, 0), 42.0);
    }

    #[test]
    fn test_nth_element_duplicates() {
        let mut data = vec![3.0, 1.0, 3.0, 1.0, 2.0];
        assert_eq!(nth_element(&mut data.clone(), 0), 1.0);
        assert_eq!(nth_element(&mut data.clone(), 1), 1.0);
        assert_eq!(nth_element(&mut data.clone(), 2), 2.0);
        assert_eq!(nth_element(&mut data.clone(), 3), 3.0);
        assert_eq!(nth_element(&mut data.clone(), 4), 3.0);
    }

    #[test]
    fn test_nth_element_nan_handling() {
        // NaN sorts as greater than everything via total_cmp
        let mut data = vec![3.0, f64::NAN, 1.0, 2.0];
        let val = nth_element(&mut data, 0);
        assert_eq!(val, 1.0);
        let mut data2 = vec![3.0, f64::NAN, 1.0, 2.0];
        let val2 = nth_element(&mut data2, 3);
        assert!(val2.is_nan());
    }

    #[test]
    fn test_median_fast_odd() {
        assert!((median_fast(&[3.0, 1.0, 2.0]).unwrap() - 2.0).abs() < 1e-15);
        assert!((median_fast(&[5.0, 1.0, 9.0, 3.0, 7.0]).unwrap() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_median_fast_even() {
        assert!((median_fast(&[4.0, 1.0, 3.0, 2.0]).unwrap() - 2.5).abs() < 1e-15);
        assert!((median_fast(&[6.0, 1.0, 5.0, 2.0, 4.0, 3.0]).unwrap() - 3.5).abs() < 1e-15);
    }

    #[test]
    fn test_median_fast_parity_with_sort_median() {
        // Verify fast median matches the sort-based median
        let data = [7.0, 2.0, 9.0, 4.0, 5.0, 1.0, 8.0, 3.0, 6.0];
        let slow = median(&data).unwrap();
        let fast = median_fast(&data).unwrap();
        assert_eq!(slow.to_bits(), fast.to_bits(), "median_fast != median for odd");

        let data2 = [7.0, 2.0, 9.0, 4.0, 5.0, 1.0, 8.0, 3.0];
        let slow2 = median(&data2).unwrap();
        let fast2 = median_fast(&data2).unwrap();
        assert_eq!(slow2.to_bits(), fast2.to_bits(), "median_fast != median for even");
    }

    #[test]
    fn test_median_fast_empty() {
        assert!(median_fast(&[]).unwrap().is_nan());
    }

    #[test]
    fn test_quantile_fast_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile_fast(&data, 0.0).unwrap() - 1.0).abs() < 1e-15);
        assert!((quantile_fast(&data, 0.5).unwrap() - 3.0).abs() < 1e-15);
        assert!((quantile_fast(&data, 1.0).unwrap() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_quantile_fast_parity_with_sort_quantile() {
        let data = [7.0, 2.0, 9.0, 4.0, 5.0, 1.0, 8.0, 3.0, 6.0];
        for p_int in 0..=20 {
            let p = p_int as f64 / 20.0;
            let slow = quantile(&data, p).unwrap();
            let fast = quantile_fast(&data, p).unwrap();
            assert!(
                (slow - fast).abs() < 1e-12,
                "quantile mismatch at p={p}: sort={slow}, fast={fast}"
            );
        }
    }

    #[test]
    fn test_sample_indices_with_replacement() {
        let idx = sample_indices(10, 5, true, 42).unwrap();
        assert_eq!(idx.len(), 5);
        assert!(idx.iter().all(|&i| i < 10));
    }

    #[test]
    fn test_sample_indices_without_replacement() {
        let idx = sample_indices(10, 5, false, 42).unwrap();
        assert_eq!(idx.len(), 5);
        assert!(idx.iter().all(|&i| i < 10));
        // All unique
        let mut sorted = idx.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
    }

    #[test]
    fn test_sample_indices_full_draw() {
        let idx = sample_indices(5, 5, false, 42).unwrap();
        assert_eq!(idx.len(), 5);
        let mut sorted = idx.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sample_indices_determinism() {
        let a = sample_indices(100, 20, false, 12345).unwrap();
        let b = sample_indices(100, 20, false, 12345).unwrap();
        assert_eq!(a, b, "same seed must produce identical indices");
    }

    #[test]
    fn test_sample_indices_error() {
        assert!(sample_indices(5, 10, false, 0).is_err());
    }

    #[test]
    fn test_sample_indices_empty() {
        assert_eq!(sample_indices(0, 5, true, 0).unwrap(), Vec::<usize>::new());
        assert_eq!(sample_indices(5, 0, true, 0).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn test_filter_mask_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [true, false, true, false, true];
        assert_eq!(filter_mask(&data, &mask).unwrap(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_filter_mask_all_true() {
        let data = [1.0, 2.0, 3.0];
        let mask = [true, true, true];
        assert_eq!(filter_mask(&data, &mask).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_filter_mask_all_false() {
        let data = [1.0, 2.0, 3.0];
        let mask = [false, false, false];
        assert_eq!(filter_mask(&data, &mask).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn test_filter_mask_length_mismatch() {
        assert!(filter_mask(&[1.0, 2.0], &[true]).is_err());
    }
}
