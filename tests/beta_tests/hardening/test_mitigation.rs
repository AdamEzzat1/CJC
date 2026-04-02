//! Hardened tests for Zero-Noise Extrapolation (ZNE).

use cjc_quantum::mitigation::*;

const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Richardson extrapolation correctness
// ---------------------------------------------------------------------------

#[test]
fn test_richardson_constant_function() {
    // Constant f(λ) = 5.0: should extrapolate to 5.0
    let result = richardson_extrapolate(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0]).unwrap();
    assert!((result.mitigated_value - 5.0).abs() < TOL);
}

#[test]
fn test_richardson_linear_function() {
    // f(λ) = 3.0 + 2.0*λ → f(0) = 3.0
    let lambdas = [1.0, 3.0];
    let values = [5.0, 9.0];
    let result = richardson_extrapolate(&lambdas, &values).unwrap();
    assert!((result.mitigated_value - 3.0).abs() < TOL,
        "linear: got {}, expected 3.0", result.mitigated_value);
}

#[test]
fn test_richardson_quadratic_function() {
    // f(λ) = 1.0 + 0.5*λ + 0.25*λ² → f(0) = 1.0
    let f = |l: f64| 1.0 + 0.5 * l + 0.25 * l * l;
    let lambdas = [1.0, 2.0, 3.0];
    let values: Vec<f64> = lambdas.iter().map(|&l| f(l)).collect();
    let result = richardson_extrapolate(&lambdas, &values).unwrap();
    assert!((result.mitigated_value - 1.0).abs() < 1e-8,
        "quadratic: got {}, expected 1.0", result.mitigated_value);
}

#[test]
fn test_richardson_cubic_function() {
    // f(λ) = 2.0 - 0.1*λ + 0.05*λ² - 0.01*λ³ → f(0) = 2.0
    let f = |l: f64| 2.0 - 0.1 * l + 0.05 * l * l - 0.01 * l * l * l;
    let lambdas = [1.0, 2.0, 3.0, 4.0];
    let values: Vec<f64> = lambdas.iter().map(|&l| f(l)).collect();
    let result = richardson_extrapolate(&lambdas, &values).unwrap();
    assert!((result.mitigated_value - 2.0).abs() < 1e-6,
        "cubic: got {}, expected 2.0", result.mitigated_value);
}

// ---------------------------------------------------------------------------
// Coefficient properties
// ---------------------------------------------------------------------------

#[test]
fn test_richardson_coefficients_sum_to_one() {
    for n in 2..6 {
        let lambdas: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let values: Vec<f64> = lambdas.iter().map(|&l| 1.0 + 0.1 * l).collect();
        let result = richardson_extrapolate(&lambdas, &values).unwrap();
        let sum: f64 = result.coefficients.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8,
            "n={}: coefficients sum to {}, expected 1.0", n, sum);
    }
}

// ---------------------------------------------------------------------------
// Linear extrapolation
// ---------------------------------------------------------------------------

#[test]
fn test_linear_extrapolate_basic() {
    // f(1) = 2, f(3) = 6 → f(0) = 0
    let result = linear_extrapolate(1.0, 2.0, 3.0, 6.0).unwrap();
    assert!((result - 0.0).abs() < TOL, "got {}", result);
}

#[test]
fn test_linear_extrapolate_negative_intercept() {
    // f(1) = -1, f(2) = -3 → slope = -2, f(0) = 1
    let result = linear_extrapolate(1.0, -1.0, 2.0, -3.0).unwrap();
    assert!((result - 1.0).abs() < TOL, "got {}", result);
}

// ---------------------------------------------------------------------------
// Noise scaling
// ---------------------------------------------------------------------------

#[test]
fn test_noise_scaling_identity_at_factor_1() {
    let p = 0.05;
    assert!((scale_depolarizing_noise(p, 1.0) - p).abs() < TOL);
    assert!((scale_dephasing_noise(p, 1.0) - p).abs() < TOL);
    assert!((scale_amplitude_damping(p, 1.0) - p).abs() < TOL);
}

#[test]
fn test_noise_scaling_zero_at_factor_0() {
    let p = 0.05;
    assert!(scale_depolarizing_noise(p, 0.0).abs() < TOL);
    assert!(scale_dephasing_noise(p, 0.0).abs() < TOL);
    assert!(scale_amplitude_damping(p, 0.0).abs() < TOL);
}

#[test]
fn test_noise_scaling_monotonically_increases() {
    let p = 0.02;
    let p1 = scale_depolarizing_noise(p, 1.0);
    let p2 = scale_depolarizing_noise(p, 2.0);
    let p3 = scale_depolarizing_noise(p, 3.0);
    assert!(p1 < p2, "p1={} < p2={}", p1, p2);
    assert!(p2 < p3, "p2={} < p3={}", p2, p3);
}

// ---------------------------------------------------------------------------
// ZNE workflow
// ---------------------------------------------------------------------------

#[test]
fn test_run_zne_recovers_noiseless_value() {
    // Simulated noisy observable: true = 1.0, noise adds 0.5*λ
    let result = run_zne(&[1.0, 2.0, 3.0], |l| 1.0 + 0.5 * l).unwrap();
    assert!((result.mitigated_value - 1.0).abs() < 1e-8,
        "ZNE: got {}, expected 1.0", result.mitigated_value);
}

#[test]
fn test_run_zne_quadratic_noise() {
    // Noisy observable with quadratic noise: true = 0.8
    let result = run_zne(&[1.0, 2.0, 3.0], |l| 0.8 + 0.1 * l * l).unwrap();
    assert!((result.mitigated_value - 0.8).abs() < 1e-6,
        "ZNE quadratic: got {}, expected 0.8", result.mitigated_value);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn test_richardson_empty_input() {
    assert!(richardson_extrapolate(&[], &[]).is_err());
}

#[test]
fn test_richardson_mismatched_lengths() {
    assert!(richardson_extrapolate(&[1.0, 2.0], &[1.0]).is_err());
}

#[test]
fn test_linear_extrapolate_same_scale() {
    assert!(linear_extrapolate(1.0, 0.5, 1.0, 0.5).is_err());
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_richardson_bitwise_deterministic() {
    let lambdas = [1.0, 1.5, 2.0, 3.0];
    let values = [0.95, 0.91, 0.85, 0.70];

    let r1 = richardson_extrapolate(&lambdas, &values).unwrap();
    let r2 = richardson_extrapolate(&lambdas, &values).unwrap();
    assert_eq!(r1.mitigated_value.to_bits(), r2.mitigated_value.to_bits(),
        "Richardson must be bit-identical");

    for (c1, c2) in r1.coefficients.iter().zip(&r2.coefficients) {
        assert_eq!(c1.to_bits(), c2.to_bits(), "coefficients must be bit-identical");
    }
}
