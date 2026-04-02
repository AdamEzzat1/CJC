//! Zero-Noise Extrapolation (ZNE) — Error Mitigation via Richardson Extrapolation.
//!
//! Mitigates noise in quantum simulations by running circuits at multiple
//! noise levels and extrapolating to the zero-noise limit.
//!
//! **Richardson Extrapolation:**
//! Given expectation values E(λ₁), E(λ₂), ..., E(λ_k) at noise scale factors
//! λ₁ < λ₂ < ... < λ_k, compute the zero-noise estimate:
//!
//!   E(0) ≈ Σᵢ cᵢ E(λᵢ)
//!
//! where cᵢ are the Richardson coefficients derived from the Vandermonde system.
//!
//! # Determinism
//!
//! - Vandermonde solve uses fixed Gaussian elimination (no pivoting needed for
//!   well-conditioned scale factors)
//! - All arithmetic uses explicit sequencing (no FMA)
//! - Kahan summation for coefficient accumulation

use cjc_repro::KahanAccumulatorF64;

// ---------------------------------------------------------------------------
// Richardson Extrapolation
// ---------------------------------------------------------------------------

/// Result of Zero-Noise Extrapolation.
#[derive(Debug, Clone)]
pub struct ZneResult {
    /// Extrapolated zero-noise expectation value.
    pub mitigated_value: f64,
    /// The noise scale factors used.
    pub scale_factors: Vec<f64>,
    /// The measured expectation values at each noise scale.
    pub measured_values: Vec<f64>,
    /// Richardson coefficients used in the extrapolation.
    pub coefficients: Vec<f64>,
}

/// Perform Richardson extrapolation to estimate the zero-noise value.
///
/// # Arguments
///
/// * `scale_factors` - Noise scale factors (e.g., [1.0, 2.0, 3.0])
/// * `measured_values` - Expectation values measured at each scale factor
///
/// # Returns
///
/// The extrapolated zero-noise estimate and diagnostic info.
///
/// # Determinism
///
/// Uses Gaussian elimination without pivoting for the Vandermonde system.
/// Kahan summation for the final weighted sum.
pub fn richardson_extrapolate(
    scale_factors: &[f64],
    measured_values: &[f64],
) -> Result<ZneResult, String> {
    let n = scale_factors.len();
    if n == 0 {
        return Err("at least one scale factor required".into());
    }
    if n != measured_values.len() {
        return Err(format!(
            "scale_factors ({}) and measured_values ({}) must have same length",
            n, measured_values.len()
        ));
    }

    // For n=1, no extrapolation needed
    if n == 1 {
        return Ok(ZneResult {
            mitigated_value: measured_values[0],
            scale_factors: scale_factors.to_vec(),
            measured_values: measured_values.to_vec(),
            coefficients: vec![1.0],
        });
    }

    // Compute Richardson coefficients via Vandermonde system.
    // We want c such that: Σ_i c_i * λ_i^k = δ_{k,0} for k = 0, 1, ..., n-1
    // This ensures the extrapolation is exact for polynomials of degree < n.
    let coefficients = compute_richardson_coefficients(scale_factors)?;

    // Compute mitigated value: E(0) = Σ_i c_i * E(λ_i)
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..n {
        acc.add(coefficients[i] * measured_values[i]);
    }

    Ok(ZneResult {
        mitigated_value: acc.finalize(),
        scale_factors: scale_factors.to_vec(),
        measured_values: measured_values.to_vec(),
        coefficients,
    })
}

/// Compute Richardson extrapolation coefficients for given scale factors.
///
/// Solves the Vandermonde system: Σ_i c_i * λ_i^k = δ_{k,0} for k=0..n-1.
/// Returns the coefficient vector c.
fn compute_richardson_coefficients(lambdas: &[f64]) -> Result<Vec<f64>, String> {
    let n = lambdas.len();

    // Build Vandermonde matrix V[k][i] = λ_i^k (transposed form)
    // and RHS b = [1, 0, 0, ..., 0]
    let mut mat = vec![vec![0.0f64; n]; n];
    let mut rhs = vec![0.0f64; n];
    rhs[0] = 1.0;

    for k in 0..n {
        for i in 0..n {
            mat[k][i] = lambdas[i].powi(k as i32);
        }
    }

    // Gaussian elimination with partial pivoting for stability
    for col in 0..n {
        // Find pivot
        let mut max_val = mat[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = mat[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return Err("Vandermonde system is singular — scale factors must be distinct".into());
        }

        // Swap rows
        if max_row != col {
            mat.swap(col, max_row);
            rhs.swap(col, max_row);
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = mat[row][col] / mat[col][col];
            for j in col..n {
                mat[row][j] -= factor * mat[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back-substitution
    let mut coefficients = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= mat[i][j] * coefficients[j];
        }
        coefficients[i] = sum / mat[i][i];
    }

    Ok(coefficients)
}

// ---------------------------------------------------------------------------
// Linear Extrapolation (simplified 2-point)
// ---------------------------------------------------------------------------

/// Simple linear extrapolation from two noise levels.
///
/// Given measurements at scale factors λ₁ and λ₂:
///   E(0) = (λ₂·E(λ₁) - λ₁·E(λ₂)) / (λ₂ - λ₁)
pub fn linear_extrapolate(
    lambda1: f64,
    value1: f64,
    lambda2: f64,
    value2: f64,
) -> Result<f64, String> {
    let denom = lambda2 - lambda1;
    if denom.abs() < 1e-15 {
        return Err("scale factors must be different for linear extrapolation".into());
    }
    Ok((lambda2 * value1 - lambda1 * value2) / denom)
}

// ---------------------------------------------------------------------------
// Noise Scaling for Density Matrix
// ---------------------------------------------------------------------------

/// Scale noise in a density matrix channel by a factor.
///
/// For depolarizing noise with parameter p, scaling by factor λ gives
/// effective noise parameter p' = 1 - (1-p)^λ.
///
/// This is used to generate the noisy measurements at different scale factors.
pub fn scale_depolarizing_noise(base_p: f64, scale_factor: f64) -> f64 {
    1.0 - (1.0 - base_p).powf(scale_factor)
}

/// Scale dephasing noise by a factor.
pub fn scale_dephasing_noise(base_p: f64, scale_factor: f64) -> f64 {
    1.0 - (1.0 - base_p).powf(scale_factor)
}

/// Scale amplitude damping noise by a factor.
pub fn scale_amplitude_damping(base_gamma: f64, scale_factor: f64) -> f64 {
    1.0 - (1.0 - base_gamma).powf(scale_factor)
}

// ---------------------------------------------------------------------------
// ZNE Workflow Helper
// ---------------------------------------------------------------------------

/// Run a Zero-Noise Extrapolation workflow.
///
/// Takes a function that computes an expectation value given a noise scale factor,
/// runs it at each scale factor, and returns the extrapolated result.
pub fn run_zne<F>(
    scale_factors: &[f64],
    compute_expectation: F,
) -> Result<ZneResult, String>
where
    F: Fn(f64) -> f64,
{
    let measured: Vec<f64> = scale_factors.iter()
        .map(|&sf| compute_expectation(sf))
        .collect();

    richardson_extrapolate(scale_factors, &measured)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_richardson_single_point() {
        let result = richardson_extrapolate(&[1.0], &[0.5]).unwrap();
        assert!((result.mitigated_value - 0.5).abs() < TOL);
    }

    #[test]
    fn test_richardson_two_point_linear() {
        // Linear function: E(λ) = 1.0 + 0.5λ
        // E(0) should be 1.0
        let lambdas = [1.0, 2.0];
        let values = [1.5, 2.0]; // 1+0.5, 1+1.0
        let result = richardson_extrapolate(&lambdas, &values).unwrap();
        assert!((result.mitigated_value - 1.0).abs() < TOL,
            "linear Richardson: got {}, expected 1.0", result.mitigated_value);
    }

    #[test]
    fn test_richardson_three_point_quadratic() {
        // Quadratic function: E(λ) = 2.0 + 0.3λ + 0.1λ²
        // E(0) should be 2.0
        let lambdas = [1.0, 2.0, 3.0];
        let values: Vec<f64> = lambdas.iter()
            .map(|&l| 2.0 + 0.3 * l + 0.1 * l * l)
            .collect();
        let result = richardson_extrapolate(&lambdas, &values).unwrap();
        assert!((result.mitigated_value - 2.0).abs() < 1e-8,
            "quadratic Richardson: got {}, expected 2.0", result.mitigated_value);
    }

    #[test]
    fn test_richardson_coefficients_sum_to_one() {
        // Richardson coefficients must sum to 1 (ensures f(0) = c·f when f is constant)
        let lambdas = [1.0, 2.0, 3.0];
        let coeffs = compute_richardson_coefficients(&lambdas).unwrap();
        let sum: f64 = coeffs.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "coefficients must sum to 1, got {}", sum);
    }

    #[test]
    fn test_linear_extrapolate() {
        // E(1) = 1.5, E(2) = 2.0 → E(0) = 1.0
        let result = linear_extrapolate(1.0, 1.5, 2.0, 2.0).unwrap();
        assert!((result - 1.0).abs() < TOL, "linear: got {}, expected 1.0", result);
    }

    #[test]
    fn test_linear_extrapolate_same_scale_error() {
        assert!(linear_extrapolate(1.0, 0.5, 1.0, 0.5).is_err());
    }

    #[test]
    fn test_richardson_mismatched_lengths() {
        assert!(richardson_extrapolate(&[1.0, 2.0], &[0.5]).is_err());
    }

    #[test]
    fn test_richardson_empty() {
        assert!(richardson_extrapolate(&[], &[]).is_err());
    }

    #[test]
    fn test_scale_depolarizing_noise() {
        // At scale_factor=1, should be same as base
        let p = 0.01;
        assert!((scale_depolarizing_noise(p, 1.0) - p).abs() < TOL);

        // At scale_factor=0, should be 0
        assert!((scale_depolarizing_noise(p, 0.0)).abs() < TOL);

        // Higher scale = more noise
        let p2 = scale_depolarizing_noise(p, 2.0);
        assert!(p2 > p, "double-scaled noise should be larger");
    }

    #[test]
    fn test_run_zne_linear_model() {
        // Simulate a noisy observable: true value = 1.0, noise adds 0.2*λ
        let result = run_zne(&[1.0, 2.0, 3.0], |lambda| 1.0 + 0.2 * lambda).unwrap();
        assert!((result.mitigated_value - 1.0).abs() < 1e-8,
            "ZNE should recover true value, got {}", result.mitigated_value);
    }

    #[test]
    fn test_richardson_determinism() {
        let lambdas = [1.0, 2.0, 3.0];
        let values = [1.3, 1.6, 1.9];
        let r1 = richardson_extrapolate(&lambdas, &values).unwrap();
        let r2 = richardson_extrapolate(&lambdas, &values).unwrap();
        assert_eq!(r1.mitigated_value.to_bits(), r2.mitigated_value.to_bits(),
            "Richardson must be bit-identical");
    }

    #[test]
    fn test_zne_result_structure() {
        let result = run_zne(&[1.0, 2.0], |l| 0.5 + 0.1 * l).unwrap();
        assert_eq!(result.scale_factors.len(), 2);
        assert_eq!(result.measured_values.len(), 2);
        assert_eq!(result.coefficients.len(), 2);
    }
}
