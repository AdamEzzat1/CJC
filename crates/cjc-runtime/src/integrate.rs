//! Numerical Integration — deterministic quadrature routines.
//!
//! # Determinism Contract
//!
//! All routines use `KahanAccumulatorF64` from `cjc_repro` for floating-point
//! summation, guaranteeing bit-identical results across runs for the same inputs.
//!
//! # Functions
//!
//! - [`trapezoid`] — composite trapezoidal rule from paired (x, y) arrays
//! - [`simpson`] — composite Simpson's 1/3 rule from paired (x, y) arrays
//! - [`cumtrapz`] — cumulative trapezoidal integration, returns array of running integrals

use cjc_repro::KahanAccumulatorF64;

// ---------------------------------------------------------------------------
// Composite Trapezoidal Rule
// ---------------------------------------------------------------------------

/// Composite trapezoidal rule for numerical integration.
///
/// Given arrays `xs` (abscissae) and `ys` (ordinates) of equal length n >= 2,
/// computes the integral approximation:
///
///   sum_{i=0}^{n-2} (xs[i+1] - xs[i]) * (ys[i] + ys[i+1]) / 2
///
/// Uses Kahan summation for the accumulation.
pub fn trapezoid(xs: &[f64], ys: &[f64]) -> Result<f64, String> {
    if xs.len() != ys.len() {
        return Err(format!(
            "trapezoid: xs and ys must have equal length, got {} and {}",
            xs.len(),
            ys.len()
        ));
    }
    if xs.len() < 2 {
        return Err("trapezoid: need at least 2 points".into());
    }

    let mut acc = KahanAccumulatorF64::new();
    for i in 0..xs.len() - 1 {
        let dx = xs[i + 1] - xs[i];
        let term = dx * (ys[i] + ys[i + 1]) * 0.5;
        acc.add(term);
    }
    Ok(acc.finalize())
}

// ---------------------------------------------------------------------------
// Composite Simpson's 1/3 Rule
// ---------------------------------------------------------------------------

/// Composite Simpson's 1/3 rule for numerical integration.
///
/// Given arrays `xs` (abscissae) and `ys` (ordinates) of equal length n >= 3,
/// where n-1 (number of subintervals) must be even, computes:
///
///   sum_{i=0,2,4,...} (xs[i+2] - xs[i]) / 6 * (ys[i] + 4*ys[i+1] + ys[i+2])
///
/// Uses Kahan summation for the accumulation.
pub fn simpson(xs: &[f64], ys: &[f64]) -> Result<f64, String> {
    if xs.len() != ys.len() {
        return Err(format!(
            "simpson: xs and ys must have equal length, got {} and {}",
            xs.len(),
            ys.len()
        ));
    }
    let n = xs.len();
    if n < 3 {
        return Err("simpson: need at least 3 points".into());
    }
    let intervals = n - 1;
    if intervals % 2 != 0 {
        return Err(format!(
            "simpson: number of subintervals must be even, got {}",
            intervals
        ));
    }

    let mut acc = KahanAccumulatorF64::new();
    let mut i = 0;
    while i + 2 < n {
        let h = (xs[i + 2] - xs[i]) / 6.0;
        let term = h * (ys[i] + 4.0 * ys[i + 1] + ys[i + 2]);
        acc.add(term);
        i += 2;
    }
    Ok(acc.finalize())
}

// ---------------------------------------------------------------------------
// Cumulative Trapezoidal Integration
// ---------------------------------------------------------------------------

/// Cumulative trapezoidal integration.
///
/// Returns an array of length n-1 where result[i] is the integral from
/// xs[0] to xs[i+1], computed via cumulative trapezoidal summation.
///
/// Uses Kahan summation for the running total.
pub fn cumtrapz(xs: &[f64], ys: &[f64]) -> Result<Vec<f64>, String> {
    if xs.len() != ys.len() {
        return Err(format!(
            "cumtrapz: xs and ys must have equal length, got {} and {}",
            xs.len(),
            ys.len()
        ));
    }
    if xs.len() < 2 {
        return Err("cumtrapz: need at least 2 points".into());
    }

    let mut acc = KahanAccumulatorF64::new();
    let mut result = Vec::with_capacity(xs.len() - 1);
    for i in 0..xs.len() - 1 {
        let dx = xs[i + 1] - xs[i];
        let term = dx * (ys[i] + ys[i + 1]) * 0.5;
        acc.add(term);
        result.push(acc.finalize());
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trapezoid_constant() {
        // Integral of f(x) = 3 from 0 to 4 = 12
        let xs: Vec<f64> = (0..=100).map(|i| i as f64 * 0.04).collect();
        let ys: Vec<f64> = xs.iter().map(|_| 3.0).collect();
        let result = trapezoid(&xs, &ys).unwrap();
        assert!((result - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_trapezoid_linear() {
        // Integral of f(x) = x from 0 to 1 = 0.5 (exact for trapezoidal)
        let n = 100;
        let xs: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.clone();
        let result = trapezoid(&xs, &ys).unwrap();
        assert!((result - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_trapezoid_sin() {
        // Integral of sin(x) from 0 to pi = 2.0
        let n = 10000;
        let xs: Vec<f64> = (0..=n)
            .map(|i| i as f64 * std::f64::consts::PI / n as f64)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        let result = trapezoid(&xs, &ys).unwrap();
        assert!(
            (result - 2.0).abs() < 1e-6,
            "trapezoid sin: expected ~2.0, got {}",
            result
        );
    }

    #[test]
    fn test_simpson_quadratic() {
        // Integral of x^2 from 0 to 1 = 1/3 (exact for Simpson on quadratics)
        let n = 100; // must be even
        let xs: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        let result = simpson(&xs, &ys).unwrap();
        assert!(
            (result - 1.0 / 3.0).abs() < 1e-10,
            "simpson x^2: expected ~0.333, got {}",
            result
        );
    }

    #[test]
    fn test_simpson_odd_intervals_error() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0];
        assert!(simpson(&xs, &ys).is_err());
    }

    #[test]
    fn test_cumtrapz_linear() {
        // Cumulative integral of f(x)=x from 0..4 with step 1
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let result = cumtrapz(&xs, &ys).unwrap();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 0.5).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
        assert!((result[2] - 4.5).abs() < 1e-12);
        assert!((result[3] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_length_mismatch_error() {
        assert!(trapezoid(&[0.0, 1.0], &[0.0]).is_err());
        assert!(simpson(&[0.0, 1.0], &[0.0]).is_err());
        assert!(cumtrapz(&[0.0, 1.0], &[0.0]).is_err());
    }

    #[test]
    fn test_determinism() {
        let n = 10000;
        let xs: Vec<f64> = (0..=n)
            .map(|i| i as f64 * std::f64::consts::PI / n as f64)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        let r1 = trapezoid(&xs, &ys).unwrap();
        let r2 = trapezoid(&xs, &ys).unwrap();
        assert_eq!(r1.to_bits(), r2.to_bits());
    }
}
