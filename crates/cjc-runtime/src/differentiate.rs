//! Numerical Differentiation — deterministic finite-difference routines.
//!
//! # Determinism Contract
//!
//! All routines produce bit-identical results for the same inputs. No
//! floating-point reductions are performed (each derivative estimate is a
//! single arithmetic expression), so no accumulator is needed.
//!
//! # Functions
//!
//! - [`diff_central`] — central difference derivative estimates at interior points
//! - [`diff_forward`] — forward difference derivative estimates
//! - [`gradient_1d`] — numerical gradient of uniformly-spaced data

// ---------------------------------------------------------------------------
// Central Difference
// ---------------------------------------------------------------------------

/// Central difference approximation of the derivative.
///
/// Given paired arrays `xs` (abscissae) and `ys` (ordinates) of length n >= 3,
/// returns n-2 derivative estimates at the interior points xs[1..n-1]:
///
///   dy/dx[i] ≈ (ys[i+1] - ys[i-1]) / (xs[i+1] - xs[i-1])
///
/// This is a second-order accurate approximation.
pub fn diff_central(xs: &[f64], ys: &[f64]) -> Result<Vec<f64>, String> {
    if xs.len() != ys.len() {
        return Err(format!(
            "diff_central: xs and ys must have equal length, got {} and {}",
            xs.len(),
            ys.len()
        ));
    }
    if xs.len() < 3 {
        return Err("diff_central: need at least 3 points".into());
    }

    let n = xs.len();
    let mut result = Vec::with_capacity(n - 2);
    for i in 1..n - 1 {
        let dx = xs[i + 1] - xs[i - 1];
        if dx == 0.0 {
            return Err(format!(
                "diff_central: zero spacing at index {}: xs[{}]={} == xs[{}]={}",
                i,
                i - 1,
                xs[i - 1],
                i + 1,
                xs[i + 1]
            ));
        }
        result.push((ys[i + 1] - ys[i - 1]) / dx);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Forward Difference
// ---------------------------------------------------------------------------

/// Forward difference approximation of the derivative.
///
/// Given paired arrays `xs` and `ys` of length n >= 2, returns n-1
/// derivative estimates:
///
///   dy/dx[i] ≈ (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])
///
/// This is a first-order accurate approximation.
pub fn diff_forward(xs: &[f64], ys: &[f64]) -> Result<Vec<f64>, String> {
    if xs.len() != ys.len() {
        return Err(format!(
            "diff_forward: xs and ys must have equal length, got {} and {}",
            xs.len(),
            ys.len()
        ));
    }
    if xs.len() < 2 {
        return Err("diff_forward: need at least 2 points".into());
    }

    let n = xs.len();
    let mut result = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = xs[i + 1] - xs[i];
        if dx == 0.0 {
            return Err(format!(
                "diff_forward: zero spacing at index {}: xs[{}]={} == xs[{}]={}",
                i, i, xs[i], i + 1, xs[i + 1]
            ));
        }
        result.push((ys[i + 1] - ys[i]) / dx);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Gradient (uniform spacing)
// ---------------------------------------------------------------------------

/// Numerical gradient of uniformly-spaced 1D data.
///
/// Given array `ys` of length n and uniform spacing `dx`, returns n
/// gradient values using:
/// - forward difference at the first point
/// - central difference at interior points
/// - backward difference at the last point
///
/// This matches NumPy's `np.gradient` behaviour for 1D uniform grids.
pub fn gradient_1d(ys: &[f64], dx: f64) -> Vec<f64> {
    let n = ys.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut result = Vec::with_capacity(n);

    // Forward difference at first point
    result.push((ys[1] - ys[0]) / dx);

    // Central difference at interior points
    let inv_2dx = 1.0 / (2.0 * dx);
    for i in 1..n - 1 {
        result.push((ys[i + 1] - ys[i - 1]) * inv_2dx);
    }

    // Backward difference at last point
    result.push((ys[n - 1] - ys[n - 2]) / dx);

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_central_quadratic() {
        // f(x) = x^2, f'(x) = 2x
        // At x=2, expect derivative ≈ 4.0
        let xs: Vec<f64> = (0..=40).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        let derivs = diff_central(&xs, &ys).unwrap();
        // Interior index for x=2.0 is i=20, but diff_central returns n-2 values
        // starting at index 1 of the original array, so derivs[19] corresponds to x=2.0
        let idx = 19; // xs[20] = 2.0, output index = 20-1 = 19
        assert!(
            (derivs[idx] - 4.0).abs() < 1e-10,
            "diff_central at x=2: expected ~4.0, got {}",
            derivs[idx]
        );
    }

    #[test]
    fn test_diff_forward_linear() {
        // f(x) = 3x + 1, f'(x) = 3
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 4.0, 7.0, 10.0, 13.0];
        let derivs = diff_forward(&xs, &ys).unwrap();
        for &d in &derivs {
            assert!((d - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gradient_1d_quadratic() {
        // f(x) = x^2 at x = 0, 1, 2, 3, 4
        let ys = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let grad = gradient_1d(&ys, 1.0);
        assert_eq!(grad.len(), 5);
        // Forward: (1 - 0)/1 = 1
        assert!((grad[0] - 1.0).abs() < 1e-12);
        // Central: (4 - 0)/2 = 2
        assert!((grad[1] - 2.0).abs() < 1e-12);
        // Central: (9 - 1)/2 = 4
        assert!((grad[2] - 4.0).abs() < 1e-12);
        // Central: (16 - 4)/2 = 6
        assert!((grad[3] - 6.0).abs() < 1e-12);
        // Backward: (16 - 9)/1 = 7
        assert!((grad[4] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_diff_central_length_mismatch() {
        assert!(diff_central(&[0.0, 1.0, 2.0], &[0.0, 1.0]).is_err());
    }

    #[test]
    fn test_diff_forward_too_few_points() {
        assert!(diff_forward(&[0.0], &[0.0]).is_err());
    }

    #[test]
    fn test_gradient_1d_empty() {
        assert!(gradient_1d(&[], 1.0).is_empty());
    }

    #[test]
    fn test_gradient_1d_single() {
        assert_eq!(gradient_1d(&[5.0], 1.0), vec![0.0]);
    }

    #[test]
    fn test_determinism() {
        let xs: Vec<f64> = (0..=100).map(|i| i as f64 * 0.01).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        let r1 = diff_central(&xs, &ys).unwrap();
        let r2 = diff_central(&xs, &ys).unwrap();
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
