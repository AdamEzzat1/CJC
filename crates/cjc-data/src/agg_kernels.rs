//! Specialized aggregate kernels for grouped operations.
//!
//! These functions operate on gathered data slices with segment boundaries,
//! or on arbitrary row-index groups. All f64 reductions use Kahan summation
//! for deterministic, numerically stable results.

use cjc_repro::kahan::KahanAccumulatorF64;
use cjc_repro::kahan_sum_f64;
use std::collections::BTreeSet;

// ── Segment-based kernels ────────────────────────────────────────────────────
// Segments are (start, end) ranges into a contiguous data slice.

/// Kahan-stable sum over contiguous segments.
pub fn agg_sum_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| kahan_sum_f64(&data[start..end]))
        .collect()
}

/// Kahan-stable mean over contiguous segments.
pub fn agg_mean_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| {
            let n = end - start;
            if n == 0 {
                return f64::NAN;
            }
            kahan_sum_f64(&data[start..end]) / n as f64
        })
        .collect()
}

/// Count per segment.
pub fn agg_count(segments: &[(usize, usize)]) -> Vec<i64> {
    segments
        .iter()
        .map(|&(start, end)| (end - start) as i64)
        .collect()
}

/// Minimum f64 per segment. Returns NAN for empty segments.
pub fn agg_min_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| {
            if start == end {
                return f64::NAN;
            }
            data[start..end]
                .iter()
                .cloned()
                .fold(f64::INFINITY, |a, b| if b.is_nan() || b < a { b } else { a })
        })
        .collect()
}

/// Maximum f64 per segment. Returns NAN for empty segments.
pub fn agg_max_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| {
            if start == end {
                return f64::NAN;
            }
            data[start..end]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, |a, b| if b.is_nan() || b > a { b } else { a })
        })
        .collect()
}

/// Variance via Welford's online algorithm (numerically stable).
/// Returns population variance (divide by N, not N-1).
pub fn agg_var_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| welford_variance(&data[start..end]))
        .collect()
}

/// Standard deviation via Welford's algorithm.
pub fn agg_sd_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| {
            let var = welford_variance(&data[start..end]);
            if var.is_nan() { f64::NAN } else { var.sqrt() }
        })
        .collect()
}

/// Median via sort per segment. Returns NAN for empty segments.
pub fn agg_median_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| {
            let n = end - start;
            if n == 0 {
                return f64::NAN;
            }
            let mut buf: Vec<f64> = data[start..end].to_vec();
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if n % 2 == 1 {
                buf[n / 2]
            } else {
                (buf[n / 2 - 1] + buf[n / 2]) / 2.0
            }
        })
        .collect()
}

/// Quantile via sort + linear interpolation. Returns NAN for empty segments.
pub fn agg_quantile_f64(data: &[f64], p: f64, segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| {
            let n = end - start;
            if n == 0 {
                return f64::NAN;
            }
            let mut buf: Vec<f64> = data[start..end].to_vec();
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p = p.clamp(0.0, 1.0);
            let idx = p * (n - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            if lo == hi {
                buf[lo]
            } else {
                let frac = idx - lo as f64;
                buf[lo] * (1.0 - frac) + buf[hi] * frac
            }
        })
        .collect()
}

/// Count distinct strings per segment using BTreeSet (deterministic).
pub fn agg_n_distinct_str(data: &[String], segments: &[(usize, usize)]) -> Vec<i64> {
    segments
        .iter()
        .map(|&(start, end)| {
            let set: BTreeSet<&String> = data[start..end].iter().collect();
            set.len() as i64
        })
        .collect()
}

/// Count distinct i64 values per segment using BTreeSet (deterministic).
pub fn agg_n_distinct_i64(data: &[i64], segments: &[(usize, usize)]) -> Vec<i64> {
    segments
        .iter()
        .map(|&(start, end)| {
            let set: BTreeSet<&i64> = data[start..end].iter().collect();
            set.len() as i64
        })
        .collect()
}

/// First f64 per segment.
pub fn agg_first_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| {
            if start == end { f64::NAN } else { data[start] }
        })
        .collect()
}

/// Last f64 per segment.
pub fn agg_last_f64(data: &[f64], segments: &[(usize, usize)]) -> Vec<f64> {
    segments
        .iter()
        .map(|&(start, end)| {
            if start == end { f64::NAN } else { data[end - 1] }
        })
        .collect()
}

/// Sum for i64 (wrapping on overflow).
pub fn agg_sum_i64(data: &[i64], segments: &[(usize, usize)]) -> Vec<i64> {
    segments
        .iter()
        .map(|&(start, end)| {
            data[start..end]
                .iter()
                .fold(0i64, |acc, &x| acc.wrapping_add(x))
        })
        .collect()
}

/// Minimum i64 per segment.
pub fn agg_min_i64(data: &[i64], segments: &[(usize, usize)]) -> Vec<i64> {
    segments
        .iter()
        .map(|&(start, end)| {
            if start == end {
                i64::MAX
            } else {
                data[start..end].iter().cloned().min().unwrap()
            }
        })
        .collect()
}

/// Maximum i64 per segment.
pub fn agg_max_i64(data: &[i64], segments: &[(usize, usize)]) -> Vec<i64> {
    segments
        .iter()
        .map(|&(start, end)| {
            if start == end {
                i64::MIN
            } else {
                data[start..end].iter().cloned().max().unwrap()
            }
        })
        .collect()
}

// ── Gather-based kernels ─────────────────────────────────────────────────────
// These work with arbitrary (non-contiguous) row indices, as produced by
// GroupIndex. They gather values first, then aggregate.

/// Kahan-stable sum over gathered f64 rows.
pub fn gather_agg_sum_f64(data: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups
        .iter()
        .map(|indices| {
            let mut acc = KahanAccumulatorF64::new();
            for &i in indices {
                acc.add(data[i]);
            }
            acc.finalize()
        })
        .collect()
}

/// Kahan-stable mean over gathered f64 rows.
pub fn gather_agg_mean_f64(data: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups
        .iter()
        .map(|indices| {
            if indices.is_empty() {
                return f64::NAN;
            }
            let mut acc = KahanAccumulatorF64::new();
            for &i in indices {
                acc.add(data[i]);
            }
            acc.finalize() / indices.len() as f64
        })
        .collect()
}

/// Welford variance over gathered f64 rows.
pub fn gather_agg_var_f64(data: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups
        .iter()
        .map(|indices| {
            let gathered: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
            welford_variance(&gathered)
        })
        .collect()
}

/// Count distinct strings over gathered rows.
pub fn gather_agg_n_distinct_str(data: &[String], groups: &[Vec<usize>]) -> Vec<i64> {
    groups
        .iter()
        .map(|indices| {
            let set: BTreeSet<&String> = indices.iter().map(|&i| &data[i]).collect();
            set.len() as i64
        })
        .collect()
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Welford's online algorithm for population variance.
/// Returns NAN for empty slices, 0.0 for single-element slices.
fn welford_variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return 0.0;
    }
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (i, &x) in values.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    m2 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_known() {
        // Variance of [2, 4, 4, 4, 5, 5, 7, 9] = 4.0 (population)
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = welford_variance(&data);
        assert!((var - 4.0).abs() < 1e-12, "got {}", var);
    }

    #[test]
    fn test_welford_empty() {
        assert!(welford_variance(&[]).is_nan());
    }

    #[test]
    fn test_welford_single() {
        assert_eq!(welford_variance(&[42.0]), 0.0);
    }
}
