//! Sliding-window functions for CJC.
//!
//! Provides `window_sum`, `window_mean`, `window_min`, `window_max` over
//! arrays of numeric values. These are commonly used in time-series analysis,
//! data science pipelines, and signal processing.
//!
//! # Determinism
//!
//! - `window_sum` and `window_mean` use Kahan summation for numerically
//!   stable, deterministic results.
//! - All window functions produce the same output for the same input on
//!   every invocation.
//!
//! # Semantics
//!
//! All window functions take `(data: &[f64], window_size: usize)` and return
//! a `Vec<f64>` of length `data.len() - window_size + 1`. That is, they
//! produce one output per valid (full) window position.
//!
//! If `window_size` is 0 or greater than `data.len()`, the result is empty.

use cjc_repro::KahanAccumulatorF64;

// ---------------------------------------------------------------------------
// Core window functions
// ---------------------------------------------------------------------------

/// Sliding-window sum with Kahan summation.
///
/// Returns a vector of length `max(0, data.len() - window_size + 1)`.
/// Each element is the sum of the corresponding window of `window_size`
/// consecutive elements.
pub fn window_sum(data: &[f64], window_size: usize) -> Vec<f64> {
    if window_size == 0 || window_size > data.len() {
        return Vec::new();
    }
    let n = data.len() - window_size + 1;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut acc = KahanAccumulatorF64::new();
        for j in 0..window_size {
            acc.add(data[i + j]);
        }
        result.push(acc.finalize());
    }

    result
}

/// Sliding-window mean with Kahan summation.
///
/// Returns a vector of length `max(0, data.len() - window_size + 1)`.
/// Each element is the arithmetic mean of the corresponding window.
pub fn window_mean(data: &[f64], window_size: usize) -> Vec<f64> {
    if window_size == 0 || window_size > data.len() {
        return Vec::new();
    }
    let n = data.len() - window_size + 1;
    let ws = window_size as f64;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut acc = KahanAccumulatorF64::new();
        for j in 0..window_size {
            acc.add(data[i + j]);
        }
        result.push(acc.finalize() / ws);
    }

    result
}

/// Sliding-window minimum.
///
/// Returns a vector of length `max(0, data.len() - window_size + 1)`.
/// Each element is the minimum of the corresponding window.
pub fn window_min(data: &[f64], window_size: usize) -> Vec<f64> {
    if window_size == 0 || window_size > data.len() {
        return Vec::new();
    }
    let n = data.len() - window_size + 1;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_val = data[i];
        for j in 1..window_size {
            let v = data[i + j];
            if v < min_val {
                min_val = v;
            }
        }
        result.push(min_val);
    }

    result
}

/// Sliding-window maximum.
///
/// Returns a vector of length `max(0, data.len() - window_size + 1)`.
/// Each element is the maximum of the corresponding window.
pub fn window_max(data: &[f64], window_size: usize) -> Vec<f64> {
    if window_size == 0 || window_size > data.len() {
        return Vec::new();
    }
    let n = data.len() - window_size + 1;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut max_val = data[i];
        for j in 1..window_size {
            let v = data[i + j];
            if v > max_val {
                max_val = v;
            }
        }
        result.push(max_val);
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_sum_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = window_sum(&data, 3);
        assert_eq!(result, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_window_sum_full() {
        let data = vec![1.0, 2.0, 3.0];
        let result = window_sum(&data, 3);
        assert_eq!(result, vec![6.0]);
    }

    #[test]
    fn test_window_sum_single() {
        let data = vec![1.0, 2.0, 3.0];
        let result = window_sum(&data, 1);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_window_sum_empty_on_too_large() {
        let data = vec![1.0, 2.0];
        let result = window_sum(&data, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_window_sum_empty_on_zero() {
        let data = vec![1.0, 2.0, 3.0];
        let result = window_sum(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_window_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = window_mean(&data, 3);
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_window_min_basic() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = window_min(&data, 3);
        assert_eq!(result, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_window_max_basic() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = window_max(&data, 3);
        assert_eq!(result, vec![4.0, 4.0, 5.0]);
    }

    #[test]
    fn test_window_determinism() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let r1 = window_sum(&data, 7);
        let r2 = window_sum(&data, 7);
        assert_eq!(r1, r2, "window_sum must be deterministic");
    }

    #[test]
    fn test_window_kahan_accuracy() {
        // Kahan summation should handle many small additions without
        // accumulating floating-point drift.
        let n = 1000;
        let data: Vec<f64> = vec![0.1; n];
        let result = window_sum(&data, n);
        assert_eq!(result.len(), 1);
        // Naive summation of 1000 * 0.1 drifts from 100.0.
        // Kahan should be very close to the true value.
        let err = (result[0] - 100.0).abs();
        assert!(
            err < 1e-12,
            "Kahan sum of 1000×0.1 should be close to 100.0, got {} (err={})",
            result[0], err,
        );
    }
}
