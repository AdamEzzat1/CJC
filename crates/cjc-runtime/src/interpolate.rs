//! Interpolation & Curve Fitting
//!
//! Provides deterministic interpolation and polynomial fitting primitives:
//! - Piecewise linear interpolation
//! - Nearest-neighbor interpolation
//! - Least-squares polynomial fitting (via Vandermonde + QR)
//! - Polynomial evaluation (Horner's method)
//! - Natural cubic spline interpolation
//!
//! # Determinism Contract
//!
//! All summation reductions use `BinnedAccumulatorF64` for bit-identical results
//! across platforms and execution orders. No `HashMap` or non-deterministic
//! iteration is used.

use crate::accumulator::BinnedAccumulatorF64;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Binary search for the interval containing `xq` in a sorted slice `xs`.
/// Returns index `i` such that `xs[i] <= xq < xs[i+1]`, clamped to valid range.
fn find_interval(xs: &[f64], xq: f64) -> usize {
    debug_assert!(xs.len() >= 2);
    if xq <= xs[0] {
        return 0;
    }
    if xq >= xs[xs.len() - 1] {
        return xs.len() - 2;
    }
    // Standard binary search for the left boundary.
    let mut lo = 0usize;
    let mut hi = xs.len() - 1;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if xs[mid] <= xq {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Validate that x_data and y_data are non-empty, same length, and x_data is
/// sorted in strictly ascending order.
fn validate_sorted_data(x_data: &[f64], y_data: &[f64]) -> Result<(), String> {
    if x_data.is_empty() {
        return Err("interpolation requires at least one data point".to_string());
    }
    if x_data.len() != y_data.len() {
        return Err(format!(
            "x_data length ({}) must equal y_data length ({})",
            x_data.len(),
            y_data.len()
        ));
    }
    if x_data.len() < 2 {
        return Err("interpolation requires at least two data points".to_string());
    }
    for i in 1..x_data.len() {
        if x_data[i] <= x_data[i - 1] {
            return Err(format!(
                "x_data must be strictly ascending; x_data[{}] = {} <= x_data[{}] = {}",
                i,
                x_data[i],
                i - 1,
                x_data[i - 1]
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. Linear Interpolation
// ---------------------------------------------------------------------------

/// Piecewise linear interpolation.
///
/// `x_data` must be sorted in strictly ascending order. Values outside the data
/// range are extrapolated as constant (clamped to the boundary y-values).
pub fn interp1d_linear(
    x_data: &[f64],
    y_data: &[f64],
    x_query: &[f64],
) -> Result<Vec<f64>, String> {
    validate_sorted_data(x_data, y_data)?;

    let n = x_data.len();
    let result: Vec<f64> = x_query
        .iter()
        .map(|&xq| {
            if xq <= x_data[0] {
                return y_data[0];
            }
            if xq >= x_data[n - 1] {
                return y_data[n - 1];
            }
            let i = find_interval(x_data, xq);
            let dx = x_data[i + 1] - x_data[i];
            if dx.abs() < 1e-300 {
                return y_data[i];
            }
            let t = (xq - x_data[i]) / dx;
            // Linear blend: y_i * (1 - t) + y_{i+1} * t
            y_data[i] + t * (y_data[i + 1] - y_data[i])
        })
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// 2. Nearest-Neighbor Interpolation
// ---------------------------------------------------------------------------

/// Nearest-neighbor interpolation.
///
/// `x_data` must be sorted in strictly ascending order. For query points exactly
/// at the midpoint between two data points, the lower-index point is chosen
/// (deterministic tie-breaking).
pub fn interp1d_nearest(
    x_data: &[f64],
    y_data: &[f64],
    x_query: &[f64],
) -> Result<Vec<f64>, String> {
    validate_sorted_data(x_data, y_data)?;

    let n = x_data.len();
    let result: Vec<f64> = x_query
        .iter()
        .map(|&xq| {
            if xq <= x_data[0] {
                return y_data[0];
            }
            if xq >= x_data[n - 1] {
                return y_data[n - 1];
            }
            let i = find_interval(x_data, xq);
            let d_left = (xq - x_data[i]).abs();
            let d_right = (x_data[i + 1] - xq).abs();
            // Tie-break: prefer lower index (<=)
            if d_left <= d_right {
                y_data[i]
            } else {
                y_data[i + 1]
            }
        })
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// 3. Polynomial Fitting (Vandermonde + QR)
// ---------------------------------------------------------------------------

/// Least-squares polynomial fit of given degree.
///
/// Returns coefficients `[a0, a1, ..., a_degree]` such that
/// `p(x) = a0 + a1*x + a2*x^2 + ... + a_degree*x^degree`.
///
/// Internally constructs the Vandermonde matrix and solves via QR decomposition
/// (Modified Gram-Schmidt) followed by back-substitution. All reductions use
/// `BinnedAccumulatorF64`.
pub fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Result<Vec<f64>, String> {
    if x.len() != y.len() {
        return Err(format!(
            "x length ({}) must equal y length ({})",
            x.len(),
            y.len()
        ));
    }
    let m = x.len(); // number of data points (rows)
    let n = degree + 1; // number of coefficients (columns)
    if m < n {
        return Err(format!(
            "need at least {} data points for degree {} fit, got {}",
            n, degree, m
        ));
    }

    // Build Vandermonde matrix V (m x n), column-major for QR convenience.
    // V[i][j] = x[i]^j
    let mut v_cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    for j in 0..n {
        let col: Vec<f64> = x
            .iter()
            .map(|&xi| {
                if j == 0 {
                    1.0
                } else {
                    // Compute xi^j iteratively (no pow for determinism)
                    let mut val = 1.0;
                    for _ in 0..j {
                        val *= xi;
                    }
                    val
                }
            })
            .collect();
        v_cols.push(col);
    }

    // QR decomposition via Modified Gram-Schmidt (column-major).
    // q_cols will hold the orthonormal columns, r is upper triangular (n x n).
    let mut q_cols = v_cols; // we modify in place
    let mut r = vec![0.0f64; n * n]; // row-major: r[i*n + j]

    for j in 0..n {
        // Orthogonalize column j against previous columns
        for i in 0..j {
            let dot = {
                let mut acc = BinnedAccumulatorF64::new();
                for k in 0..m {
                    acc.add(q_cols[i][k] * q_cols[j][k]);
                }
                acc.finalize()
            };
            r[i * n + j] = dot;
            for k in 0..m {
                q_cols[j][k] -= dot * q_cols[i][k];
            }
        }
        // Compute norm of column j
        let norm = {
            let mut acc = BinnedAccumulatorF64::new();
            for k in 0..m {
                acc.add(q_cols[j][k] * q_cols[j][k]);
            }
            acc.finalize()
        }
        .sqrt();

        r[j * n + j] = norm;
        if norm < 1e-15 {
            return Err("polyfit: Vandermonde matrix is rank-deficient".to_string());
        }
        for k in 0..m {
            q_cols[j][k] /= norm;
        }
    }

    // Compute Q^T * y
    let mut qty = vec![0.0f64; n];
    for j in 0..n {
        let mut acc = BinnedAccumulatorF64::new();
        for k in 0..m {
            acc.add(q_cols[j][k] * y[k]);
        }
        qty[j] = acc.finalize();
    }

    // Back-substitution: R * coeffs = Q^T * y
    let mut coeffs = vec![0.0f64; n];
    for j in (0..n).rev() {
        let mut acc = BinnedAccumulatorF64::new();
        for k in (j + 1)..n {
            acc.add(r[j * n + k] * coeffs[k]);
        }
        coeffs[j] = (qty[j] - acc.finalize()) / r[j * n + j];
    }

    Ok(coeffs)
}

// ---------------------------------------------------------------------------
// 4. Polynomial Evaluation (Horner's Method)
// ---------------------------------------------------------------------------

/// Evaluate polynomial at given points using Horner's method.
///
/// `coeffs` = `[a0, a1, ..., an]` where `p(x) = a0 + a1*x + ... + an*x^n`.
/// Horner form: `p(x) = a0 + x*(a1 + x*(a2 + ... + x*an))`.
pub fn polyval(coeffs: &[f64], x: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0; x.len()];
    }
    x.iter()
        .map(|&xi| {
            // Horner's: start from highest degree coefficient
            let mut result = coeffs[coeffs.len() - 1];
            for k in (0..coeffs.len() - 1).rev() {
                result = result * xi + coeffs[k];
            }
            result
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 5. Cubic Spline
// ---------------------------------------------------------------------------

/// Natural cubic spline representation.
///
/// For each interval `[x[i], x[i+1]]` the spline is:
/// ```text
/// S_i(t) = a[i] + b[i]*t + c[i]*t^2 + d[i]*t^3
/// ```
/// where `t = x_query - x[i]`.
pub struct CubicSpline {
    /// Knot positions (sorted, length n).
    pub x: Vec<f64>,
    /// Constant coefficients (= y values at knots), length n-1.
    pub a: Vec<f64>,
    /// Linear coefficients, length n-1.
    pub b: Vec<f64>,
    /// Quadratic coefficients, length n-1.
    pub c: Vec<f64>,
    /// Cubic coefficients, length n-1.
    pub d: Vec<f64>,
}

/// Construct a natural cubic spline (second derivative = 0 at boundaries).
///
/// Solves the tridiagonal system for the second-derivative values using the
/// Thomas algorithm. All reductions use `BinnedAccumulatorF64`.
pub fn spline_cubic_natural(x: &[f64], y: &[f64]) -> Result<CubicSpline, String> {
    validate_sorted_data(x, y)?;

    let n = x.len(); // number of data points
    let nm1 = n - 1; // number of intervals

    // Step 1: Compute interval widths h[i] and divided differences
    let h: Vec<f64> = (0..nm1).map(|i| x[i + 1] - x[i]).collect();

    // Step 2: Set up the tridiagonal system for second derivatives (c_i).
    // For natural spline: c[0] = 0, c[n-1] = 0.
    // Interior equations (i = 1..n-2):
    //   h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1]
    //     = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    // We only solve for c[1..n-2] (the interior points).
    // With natural BC: c[0] = 0 and c[n-1] = 0.

    let mut c_all = vec![0.0f64; n]; // second-derivative-like values

    if n > 2 {
        let interior = n - 2; // number of interior unknowns

        // Build tridiagonal: lower (a_tri), diagonal (b_tri), upper (c_tri), rhs (d_tri)
        let mut a_tri = vec![0.0f64; interior]; // sub-diagonal
        let mut b_tri = vec![0.0f64; interior]; // main diagonal
        let mut c_tri = vec![0.0f64; interior]; // super-diagonal
        let mut d_tri = vec![0.0f64; interior]; // right-hand side

        for i in 0..interior {
            let ii = i + 1; // index into original arrays
            a_tri[i] = h[ii - 1];
            b_tri[i] = 2.0 * (h[ii - 1] + h[ii]);
            c_tri[i] = h[ii];
            d_tri[i] = 3.0 * ((y[ii + 1] - y[ii]) / h[ii] - (y[ii] - y[ii - 1]) / h[ii - 1]);
        }

        // Thomas algorithm (tridiagonal solver)
        // Forward sweep
        for i in 1..interior {
            if b_tri[i - 1].abs() < 1e-300 {
                return Err("spline_cubic_natural: zero pivot in Thomas algorithm".to_string());
            }
            let w = a_tri[i] / b_tri[i - 1];
            b_tri[i] -= w * c_tri[i - 1];
            d_tri[i] -= w * d_tri[i - 1];
        }

        // Back substitution
        if b_tri[interior - 1].abs() < 1e-300 {
            return Err("spline_cubic_natural: zero pivot in Thomas algorithm".to_string());
        }
        c_all[interior] = d_tri[interior - 1] / b_tri[interior - 1]; // c_all[n-2]
        for i in (0..interior - 1).rev() {
            c_all[i + 1] = (d_tri[i] - c_tri[i] * c_all[i + 2]) / b_tri[i];
        }
    }
    // c_all[0] = 0 and c_all[n-1] = 0 (natural boundary conditions)

    // Step 3: Compute spline coefficients for each interval.
    let mut a_coeff = Vec::with_capacity(nm1);
    let mut b_coeff = Vec::with_capacity(nm1);
    let mut c_coeff = Vec::with_capacity(nm1);
    let mut d_coeff = Vec::with_capacity(nm1);

    for i in 0..nm1 {
        a_coeff.push(y[i]);
        b_coeff.push(
            (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c_all[i] + c_all[i + 1]) / 3.0,
        );
        c_coeff.push(c_all[i]);
        d_coeff.push((c_all[i + 1] - c_all[i]) / (3.0 * h[i]));
    }

    Ok(CubicSpline {
        x: x.to_vec(),
        a: a_coeff,
        b: b_coeff,
        c: c_coeff,
        d: d_coeff,
    })
}

/// Evaluate a cubic spline at query points.
///
/// Uses binary search to locate the interval for each query point. Points
/// outside the data range are clamped to the boundary intervals.
pub fn spline_eval(spline: &CubicSpline, x_query: &[f64]) -> Result<Vec<f64>, String> {
    if spline.x.len() < 2 {
        return Err("spline must have at least two knots".to_string());
    }

    let result: Vec<f64> = x_query
        .iter()
        .map(|&xq| {
            let i = find_interval(&spline.x, xq);
            let t = xq - spline.x[i];
            // Horner-like evaluation: a + t*(b + t*(c + t*d))
            spline.a[i] + t * (spline.b[i] + t * (spline.c[i] + t * spline.d[i]))
        })
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod interpolate_tests {
    use super::*;

    const EPS: f64 = 1e-12;

    // --- Linear Interpolation ---

    #[test]
    fn linear_interp_exact_at_data_points() {
        // y = 2x + 1 at x = [0, 1, 2, 3, 4]
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();
        let result = interp1d_linear(&x_data, &y_data, &x_data).unwrap();
        for (i, &r) in result.iter().enumerate() {
            assert!(
                (r - y_data[i]).abs() < EPS,
                "at x={}, expected {}, got {}",
                x_data[i],
                y_data[i],
                r
            );
        }
    }

    #[test]
    fn linear_interp_between_points() {
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();
        let x_query = vec![0.5, 1.5, 2.5, 3.5];
        let result = interp1d_linear(&x_data, &y_data, &x_query).unwrap();
        for (i, &xq) in x_query.iter().enumerate() {
            let expected = 2.0 * xq + 1.0;
            assert!(
                (result[i] - expected).abs() < EPS,
                "at x={}, expected {}, got {}",
                xq,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn linear_interp_extrapolation_clamps() {
        let x_data = vec![1.0, 2.0, 3.0];
        let y_data = vec![10.0, 20.0, 30.0];
        let x_query = vec![-5.0, 0.0, 4.0, 100.0];
        let result = interp1d_linear(&x_data, &y_data, &x_query).unwrap();
        assert!((result[0] - 10.0).abs() < EPS); // clamp left
        assert!((result[1] - 10.0).abs() < EPS); // clamp left
        assert!((result[2] - 30.0).abs() < EPS); // clamp right
        assert!((result[3] - 30.0).abs() < EPS); // clamp right
    }

    #[test]
    fn linear_interp_validation_errors() {
        assert!(interp1d_linear(&[], &[], &[1.0]).is_err());
        assert!(interp1d_linear(&[1.0], &[1.0], &[1.0]).is_err());
        assert!(interp1d_linear(&[1.0, 2.0], &[1.0], &[1.0]).is_err()); // length mismatch
        assert!(interp1d_linear(&[2.0, 1.0], &[1.0, 2.0], &[1.0]).is_err()); // not sorted
    }

    // --- Nearest-Neighbor Interpolation ---

    #[test]
    fn nearest_interp_snaps_to_closest() {
        let x_data = vec![0.0, 1.0, 3.0, 5.0];
        let y_data = vec![10.0, 20.0, 30.0, 40.0];
        // 0.3 is closer to 0.0 -> 10.0
        // 0.7 is closer to 1.0 -> 20.0
        // 2.5 is closer to 3.0 -> 30.0
        // 4.0 is equidistant between 3.0 and 5.0 -> tie-break to 30.0 (lower index)
        let x_query = vec![0.3, 0.7, 2.5, 4.0];
        let result = interp1d_nearest(&x_data, &y_data, &x_query).unwrap();
        assert!((result[0] - 10.0).abs() < EPS);
        assert!((result[1] - 20.0).abs() < EPS);
        assert!((result[2] - 30.0).abs() < EPS);
        assert!((result[3] - 30.0).abs() < EPS); // tie-break: lower index
    }

    #[test]
    fn nearest_interp_at_data_points() {
        let x_data = vec![0.0, 1.0, 2.0];
        let y_data = vec![5.0, 15.0, 25.0];
        let result = interp1d_nearest(&x_data, &y_data, &x_data).unwrap();
        for (i, &r) in result.iter().enumerate() {
            assert!((r - y_data[i]).abs() < EPS);
        }
    }

    #[test]
    fn nearest_interp_boundary_clamp() {
        let x_data = vec![1.0, 2.0, 3.0];
        let y_data = vec![10.0, 20.0, 30.0];
        let result = interp1d_nearest(&x_data, &y_data, &[-1.0, 0.5, 3.5, 100.0]).unwrap();
        assert!((result[0] - 10.0).abs() < EPS);
        assert!((result[1] - 10.0).abs() < EPS);
        assert!((result[2] - 30.0).abs() < EPS);
        assert!((result[3] - 30.0).abs() < EPS);
    }

    // --- Polynomial Fitting ---

    #[test]
    fn polyfit_degree1_linear_data() {
        // y = 3x + 2
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();
        let coeffs = polyfit(&x, &y, 1).unwrap();
        assert_eq!(coeffs.len(), 2);
        assert!(
            (coeffs[0] - 2.0).abs() < 1e-10,
            "intercept: expected 2.0, got {}",
            coeffs[0]
        );
        assert!(
            (coeffs[1] - 3.0).abs() < 1e-10,
            "slope: expected 3.0, got {}",
            coeffs[1]
        );
    }

    #[test]
    fn polyfit_degree2_quadratic_data() {
        // y = 1 + 2x + 3x^2
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + 3.0 * xi * xi).collect();
        let coeffs = polyfit(&x, &y, 2).unwrap();
        assert_eq!(coeffs.len(), 3);
        assert!(
            (coeffs[0] - 1.0).abs() < 1e-8,
            "a0: expected 1.0, got {}",
            coeffs[0]
        );
        assert!(
            (coeffs[1] - 2.0).abs() < 1e-8,
            "a1: expected 2.0, got {}",
            coeffs[1]
        );
        assert!(
            (coeffs[2] - 3.0).abs() < 1e-8,
            "a2: expected 3.0, got {}",
            coeffs[2]
        );
    }

    #[test]
    fn polyfit_validation_errors() {
        assert!(polyfit(&[1.0], &[1.0, 2.0], 1).is_err()); // length mismatch
        assert!(polyfit(&[1.0], &[1.0], 1).is_err()); // too few points for degree 1
    }

    // --- Polynomial Evaluation ---

    #[test]
    fn polyval_roundtrip_with_polyfit() {
        // Fit y = 5 - x + 0.5*x^2 and evaluate at same points
        let x: Vec<f64> = (0..15).map(|i| i as f64 * 0.3).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 5.0 - xi + 0.5 * xi * xi).collect();
        let coeffs = polyfit(&x, &y, 2).unwrap();
        let y_eval = polyval(&coeffs, &x);
        for (i, (&ye, &yo)) in y_eval.iter().zip(y.iter()).enumerate() {
            assert!(
                (ye - yo).abs() < 1e-8,
                "at i={}, expected {}, got {}",
                i,
                yo,
                ye
            );
        }
    }

    #[test]
    fn polyval_empty_coeffs() {
        let result = polyval(&[], &[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3);
        for &r in &result {
            assert!((r - 0.0).abs() < EPS);
        }
    }

    #[test]
    fn polyval_constant() {
        let result = polyval(&[42.0], &[0.0, 1.0, 100.0]);
        for &r in &result {
            assert!((r - 42.0).abs() < EPS);
        }
    }

    // --- Cubic Spline ---

    #[test]
    fn spline_interpolates_data_points_exactly() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let spline = spline_cubic_natural(&x, &y).unwrap();
        let result = spline_eval(&spline, &x).unwrap();
        for (i, (&r, &yi)) in result.iter().zip(y.iter()).enumerate() {
            assert!(
                (r - yi).abs() < EPS,
                "at knot x={}, expected {}, got {}",
                x[i],
                yi,
                r
            );
        }
    }

    #[test]
    fn spline_linear_data_is_exact() {
        // For linear data y = 2x + 1, natural cubic spline should reproduce it exactly
        // (since a cubic with zero higher-order terms IS a line).
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let spline = spline_cubic_natural(&x, &y).unwrap();
        let x_query = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let result = spline_eval(&spline, &x_query).unwrap();
        for (i, &xq) in x_query.iter().enumerate() {
            let expected = 2.0 * xq + 1.0;
            assert!(
                (result[i] - expected).abs() < 1e-10,
                "at x={}, expected {}, got {}",
                xq,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn spline_derivative_continuity_at_knots() {
        // Numerically verify that the first derivative is continuous at interior knots.
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 2.0, 1.0, 3.0, 0.5];
        let spline = spline_cubic_natural(&x, &y).unwrap();

        let delta = 1e-7;
        // Check derivative continuity at interior knots (indices 1, 2, 3)
        for &knot in &[1.0, 2.0, 3.0] {
            let left = spline_eval(&spline, &[knot - delta]).unwrap()[0];
            let right = spline_eval(&spline, &[knot + delta]).unwrap()[0];
            let at = spline_eval(&spline, &[knot]).unwrap()[0];
            let deriv_left = (at - left) / delta;
            let deriv_right = (right - at) / delta;
            let diff = (deriv_left - deriv_right).abs();
            assert!(
                diff < 1e-4,
                "derivative discontinuity at x={}: left={}, right={}, diff={}",
                knot,
                deriv_left,
                deriv_right,
                diff
            );
        }
    }

    #[test]
    fn spline_second_derivative_zero_at_boundaries() {
        // Natural spline: second derivative is zero at x[0] and x[n-1].
        // S''(x) = 2*c + 6*d*t
        // At left boundary (t=0): S''(x[0]) = 2*c[0] should be ~0
        // At right boundary: evaluate in last interval at t = x[n-1] - x[n-2]
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let spline = spline_cubic_natural(&x, &y).unwrap();

        // Left boundary
        assert!(
            (2.0 * spline.c[0]).abs() < 1e-10,
            "second derivative at left boundary = {}",
            2.0 * spline.c[0]
        );

        // Right boundary: 2*c[n-2] + 6*d[n-2]*h[n-2]
        let last = spline.a.len() - 1;
        let h_last = x[last + 1] - x[last];
        let second_deriv_right = 2.0 * spline.c[last] + 6.0 * spline.d[last] * h_last;
        assert!(
            second_deriv_right.abs() < 1e-10,
            "second derivative at right boundary = {}",
            second_deriv_right
        );
    }

    #[test]
    fn spline_validation_errors() {
        assert!(spline_cubic_natural(&[], &[]).is_err());
        assert!(spline_cubic_natural(&[1.0], &[1.0]).is_err());
        assert!(spline_cubic_natural(&[2.0, 1.0], &[1.0, 2.0]).is_err()); // not sorted
    }

    // --- Determinism Tests ---

    #[test]
    fn determinism_linear_interp() {
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![0.0, 0.8, 0.9, 0.1, -0.8, -1.0];
        let x_query: Vec<f64> = (0..100).map(|i| i as f64 * 0.05).collect();
        let r1 = interp1d_linear(&x_data, &y_data, &x_query).unwrap();
        let r2 = interp1d_linear(&x_data, &y_data, &x_query).unwrap();
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic result detected");
        }
    }

    #[test]
    fn determinism_polyfit() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let c1 = polyfit(&x, &y, 5).unwrap();
        let c2 = polyfit(&x, &y, 5).unwrap();
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic polyfit");
        }
    }

    #[test]
    fn determinism_cubic_spline() {
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let y = vec![0.0, 0.48, 0.84, 1.0, 0.91, 0.60, 0.14];
        let x_query: Vec<f64> = (0..60).map(|i| i as f64 * 0.05).collect();
        let s1 = spline_cubic_natural(&x, &y).unwrap();
        let s2 = spline_cubic_natural(&x, &y).unwrap();
        let r1 = spline_eval(&s1, &x_query).unwrap();
        let r2 = spline_eval(&s2, &x_query).unwrap();
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic spline");
        }
    }
}
