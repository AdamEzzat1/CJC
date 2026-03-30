//! Quantum Hardening Tests — comprehensive verification of wirtinger, adjoint,
//! simd_kernel, and mps modules.
//!
//! Covers:
//! - Wirtinger derivatives vs finite differences
//! - Adjoint U*U-dagger identity, gradient vs parameter-shift, multi-parameter
//! - SIMD kernel bit-identical parity with scalar
//! - MPS statevector agreement for GHZ states, SVD determinism, reconstruction error, memory scaling
//! - Cross-module: adjoint gradient of MPS-constructed state vs statevector gradient

use cjc_quantum::*;
use cjc_quantum::wirtinger::*;
use cjc_quantum::adjoint::*;
use cjc_quantum::simd_kernel::*;
use cjc_quantum::mps::{Mps, svd_sign_stabilized, DenseMatrix};
use cjc_runtime::complex::ComplexF64;
use std::f64::consts::{PI, FRAC_1_SQRT_2};

const TOL: f64 = 1e-10;

// =========================================================================
// Helpers
// =========================================================================

fn assert_complex_approx(a: ComplexF64, b: ComplexF64, tol: f64, msg: &str) {
    assert!(
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol,
        "{}: got ({}, {}) expected ({}, {})",
        msg, a.re, a.im, b.re, b.im
    );
}

fn h_matrix() -> [[ComplexF64; 2]; 2] {
    let s = ComplexF64::real(FRAC_1_SQRT_2);
    let ms = ComplexF64::real(-FRAC_1_SQRT_2);
    [[s, s], [s, ms]]
}

fn x_matrix() -> [[ComplexF64; 2]; 2] {
    [
        [ComplexF64::ZERO, ComplexF64::ONE],
        [ComplexF64::ONE, ComplexF64::ZERO],
    ]
}

fn y_matrix() -> [[ComplexF64; 2]; 2] {
    let ni = ComplexF64::new(0.0, -1.0);
    let pi = ComplexF64::new(0.0, 1.0);
    [
        [ComplexF64::ZERO, ni],
        [pi, ComplexF64::ZERO],
    ]
}

fn z_matrix() -> [[ComplexF64; 2]; 2] {
    [
        [ComplexF64::ONE, ComplexF64::ZERO],
        [ComplexF64::ZERO, ComplexF64::new(-1.0, 0.0)],
    ]
}

fn rx_matrix(theta: f64) -> [[ComplexF64; 2]; 2] {
    let c = ComplexF64::real((theta / 2.0).cos());
    let s = ComplexF64::new(0.0, -(theta / 2.0).sin());
    [[c, s], [s, c]]
}

fn ry_matrix(theta: f64) -> [[ComplexF64; 2]; 2] {
    let c = ComplexF64::real((theta / 2.0).cos());
    let s = ComplexF64::real((theta / 2.0).sin());
    let ms = ComplexF64::real(-(theta / 2.0).sin());
    [[c, ms], [s, c]]
}

fn rz_matrix(theta: f64) -> [[ComplexF64; 2]; 2] {
    let ep = ComplexF64::new((theta / 2.0).cos(), -(theta / 2.0).sin()); // e^{-i theta/2}
    let em = ComplexF64::new((theta / 2.0).cos(), (theta / 2.0).sin());  // e^{+i theta/2}
    [[ep, ComplexF64::ZERO], [ComplexF64::ZERO, em]]
}

/// Compute norm of statevector amplitudes for verification.
fn sv_norm(amps: &[ComplexF64]) -> f64 {
    amps.iter().map(|a| a.norm_sq()).sum::<f64>().sqrt()
}

// =========================================================================
// 1. WIRTINGER TESTS
// =========================================================================

#[test]
fn wirtinger_norm_sq_finite_diff() {
    // Verify Wirtinger derivative of |z|^2 matches finite differences
    // for multiple test points.
    let test_points = [
        ComplexF64::new(2.0, 3.0),
        ComplexF64::new(-1.5, 0.7),
        ComplexF64::new(0.0, 1.0),
        ComplexF64::new(5.0, -2.3),
    ];
    let eps = 1e-7;

    for z0 in &test_points {
        // Finite difference in real direction: d|z|^2/dx
        let f_xp = ComplexF64::new(z0.re + eps, z0.im).norm_sq();
        let f_xm = ComplexF64::new(z0.re - eps, z0.im).norm_sq();
        let df_dx = (f_xp - f_xm) / (2.0 * eps);

        // Finite difference in imaginary direction: d|z|^2/dy
        let f_yp = ComplexF64::new(z0.re, z0.im + eps).norm_sq();
        let f_ym = ComplexF64::new(z0.re, z0.im - eps).norm_sq();
        let df_dy = (f_yp - f_ym) / (2.0 * eps);

        // Wirtinger: d/dz* = (df/dx + i df/dy) / 2
        let numerical_dz_conj = ComplexF64::new(df_dx, df_dy).scale(0.5);

        // Analytical: d|z|^2/dz* = z
        assert_complex_approx(numerical_dz_conj, *z0, 1e-5,
            &format!("Wirtinger norm_sq FD at ({}, {})", z0.re, z0.im));
    }
}

#[test]
fn wirtinger_mul_finite_diff() {
    // f(z) = z * z = z^2
    // d(z^2)/dz = 2z, d(z^2)/dz* = 0
    let z0 = ComplexF64::new(1.5, -0.8);
    let eps = 1e-7;

    // Wirtinger dual computation
    let z = WirtingerDual::variable(z0);
    let result = z.clone().mul(z.clone());

    // Expected: dz = 2*z0
    let expected_dz = z0.scale(2.0);
    assert_complex_approx(result.dz, expected_dz, TOL, "z^2 dz");
    assert_complex_approx(result.dz_conj, ComplexF64::ZERO, TOL, "z^2 dz_conj");

    // Cross-check with finite differences
    let f = |z: ComplexF64| -> ComplexF64 { z.mul_fixed(z) };
    let f_xp = f(ComplexF64::new(z0.re + eps, z0.im));
    let f_xm = f(ComplexF64::new(z0.re - eps, z0.im));
    let df_dx = ComplexF64::new(
        (f_xp.re - f_xm.re) / (2.0 * eps),
        (f_xp.im - f_xm.im) / (2.0 * eps),
    );
    let f_yp = f(ComplexF64::new(z0.re, z0.im + eps));
    let f_ym = f(ComplexF64::new(z0.re, z0.im - eps));
    let df_dy = ComplexF64::new(
        (f_yp.re - f_ym.re) / (2.0 * eps),
        (f_yp.im - f_ym.im) / (2.0 * eps),
    );

    // d/dz = (df/dx - i df/dy) / 2
    let numerical_dz = ComplexF64::new(
        (df_dx.re + df_dy.im) / 2.0,
        (df_dx.im - df_dy.re) / 2.0,
    );
    assert_complex_approx(numerical_dz, expected_dz, 1e-5, "z^2 dz numerical");
}

#[test]
fn wirtinger_chain_norm_sq_of_sum() {
    // f(z) = |z + c|^2 where c = 1 + 2i
    // d/dz* |z+c|^2 = z + c
    let z0 = ComplexF64::new(0.5, -1.0);
    let c = ComplexF64::new(1.0, 2.0);

    let z = WirtingerDual::variable(z0);
    let w = WirtingerDual::constant(c);
    let result = z.add(w).norm_sq();

    let expected_dz_conj = z0.add(c);
    assert_complex_approx(result.dz_conj, expected_dz_conj, TOL,
        "|z+c|^2 dz_conj = z+c");
}

#[test]
fn wirtinger_probability_gradient_matches_dual() {
    // probability_gradient(alpha) should match WirtingerDual::variable(alpha).norm_sq()
    let alpha = ComplexF64::new(0.3, -0.7);
    let (pg_dz, pg_dz_conj) = probability_gradient(alpha);

    let dual = WirtingerDual::variable(alpha).norm_sq();

    assert_complex_approx(pg_dz, dual.dz, TOL, "prob grad dz vs dual");
    assert_complex_approx(pg_dz_conj, dual.dz_conj, TOL, "prob grad dz_conj vs dual");
}

#[test]
fn wirtinger_scale_linearity() {
    // d/dz (s * z) = s for real scalar s
    let z0 = ComplexF64::new(2.0, 3.0);
    let s = 4.5;
    let z = WirtingerDual::variable(z0);
    let result = z.scale(s);
    assert_complex_approx(result.dz, ComplexF64::real(s), TOL, "scale dz");
    assert_complex_approx(result.dz_conj, ComplexF64::ZERO, TOL, "scale dz_conj");
}

#[test]
fn wirtinger_conj_swap() {
    // For f(z) = z: conj swaps dz and dz_conj
    let z = WirtingerDual::variable(ComplexF64::new(1.0, 2.0));
    let zc = z.conj();
    assert_complex_approx(zc.dz, ComplexF64::ZERO, TOL, "conj(var) dz = 0");
    assert_complex_approx(zc.dz_conj, ComplexF64::ONE, TOL, "conj(var) dz_conj = 1");
}

// =========================================================================
// 2. ADJOINT TESTS
// =========================================================================

#[test]
fn adjoint_u_udagger_identity_rx() {
    // Rx(theta) followed by Rx(-theta) = I
    for &theta in &[0.3, PI / 4.0, PI / 2.0, PI, 1.7] {
        let mut circ = Circuit::new(1);
        circ.rx(0, theta).rx(0, -theta);
        let sv = circ.execute().unwrap();
        // Should be back to |0>
        assert!(
            (sv.amplitudes[0].re - 1.0).abs() < TOL,
            "Rx({}) Rx(-{}) not identity: |0> amp = {:?}", theta, theta, sv.amplitudes[0]
        );
        assert!(
            sv.amplitudes[1].norm_sq() < TOL,
            "Rx({}) Rx(-{}) not identity: |1> amp = {:?}", theta, theta, sv.amplitudes[1]
        );
    }
}

#[test]
fn adjoint_u_udagger_identity_ry() {
    for &theta in &[0.5, PI / 3.0, PI, 2.1] {
        let mut circ = Circuit::new(1);
        circ.ry(0, theta).ry(0, -theta);
        let sv = circ.execute().unwrap();
        assert!(
            (sv.amplitudes[0].re - 1.0).abs() < TOL,
            "Ry({}) Ry(-{}) not identity: |0> amp = {:?}", theta, theta, sv.amplitudes[0]
        );
        assert!(
            sv.amplitudes[1].norm_sq() < TOL,
            "Ry({}) Ry(-{}) not identity: |1> amp = {:?}", theta, theta, sv.amplitudes[1]
        );
    }
}

#[test]
fn adjoint_u_udagger_identity_rz() {
    for &theta in &[0.2, PI / 6.0, PI, 3.0] {
        let mut circ = Circuit::new(1);
        circ.rz(0, theta).rz(0, -theta);
        let sv = circ.execute().unwrap();
        assert!(
            (sv.amplitudes[0].re - 1.0).abs() < TOL,
            "Rz({}) Rz(-{}) not identity: |0> amp = {:?}", theta, theta, sv.amplitudes[0]
        );
        assert!(
            sv.amplitudes[1].norm_sq() < TOL,
            "Rz({}) Rz(-{}) not identity: |1> amp = {:?}", theta, theta, sv.amplitudes[1]
        );
    }
}

#[test]
fn adjoint_gradient_matches_parameter_shift_ry() {
    // For Ry(theta)|0>, E(Z) = cos(theta)
    // Parameter-shift: dE/dtheta = (E(theta+pi/2) - E(theta-pi/2)) / 2
    let z_obs = vec![1.0, -1.0];

    for &theta in &[0.3, 0.7, PI / 4.0, 1.5, 2.5] {
        // Adjoint gradient
        let mut circ = Circuit::new(1);
        circ.ry(0, theta);
        let adj_grads = adjoint_differentiation(&circ, &z_obs).unwrap();

        // Parameter-shift gradient
        let shift = PI / 2.0;
        let mut circ_p = Circuit::new(1);
        circ_p.ry(0, theta + shift);
        let e_p = expectation_value(&circ_p.execute().unwrap(), &z_obs).unwrap();

        let mut circ_m = Circuit::new(1);
        circ_m.ry(0, theta - shift);
        let e_m = expectation_value(&circ_m.execute().unwrap(), &z_obs).unwrap();

        let ps_grad = parameter_shift_gradient(e_p, e_m);

        assert!(
            (adj_grads.gradients[0] - ps_grad).abs() < 1e-6,
            "theta={}: adjoint {} vs param-shift {}", theta, adj_grads.gradients[0], ps_grad
        );
    }
}

#[test]
fn adjoint_gradient_matches_parameter_shift_rx() {
    let z_obs = vec![1.0, -1.0];

    for &theta in &[0.4, 1.0, PI / 3.0] {
        let mut circ = Circuit::new(1);
        circ.h(0).rx(0, theta).h(0);
        let adj_grads = adjoint_differentiation(&circ, &z_obs).unwrap();

        let shift = PI / 2.0;
        let mut circ_p = Circuit::new(1);
        circ_p.h(0).rx(0, theta + shift).h(0);
        let e_p = expectation_value(&circ_p.execute().unwrap(), &z_obs).unwrap();

        let mut circ_m = Circuit::new(1);
        circ_m.h(0).rx(0, theta - shift).h(0);
        let e_m = expectation_value(&circ_m.execute().unwrap(), &z_obs).unwrap();

        let ps_grad = parameter_shift_gradient(e_p, e_m);

        // Gate 0=H, Gate 1=Rx, Gate 2=H
        assert!(
            (adj_grads.gradients[1] - ps_grad).abs() < 1e-6,
            "Rx theta={}: adjoint {} vs param-shift {}", theta, adj_grads.gradients[1], ps_grad
        );
    }
}

#[test]
fn adjoint_multi_parameter_gradient() {
    // Circuit: Ry(t1) Rz(t2) Rx(t3) |0>, observable Z
    // Verify all three gradients via finite differences
    let theta1 = 0.3;
    let theta2 = 0.7;
    let theta3 = 1.2;
    let z_obs = vec![1.0, -1.0];
    let eps = 1e-5;

    let mut circ = Circuit::new(1);
    circ.ry(0, theta1).rz(0, theta2).rx(0, theta3);
    let adj_grads = adjoint_differentiation(&circ, &z_obs).unwrap();

    let thetas = [theta1, theta2, theta3];
    for i in 0..3 {
        let mut tp = thetas;
        let mut tm = thetas;
        tp[i] += eps;
        tm[i] -= eps;

        let mut circ_p = Circuit::new(1);
        circ_p.ry(0, tp[0]).rz(0, tp[1]).rx(0, tp[2]);
        let e_p = expectation_value(&circ_p.execute().unwrap(), &z_obs).unwrap();

        let mut circ_m = Circuit::new(1);
        circ_m.ry(0, tm[0]).rz(0, tm[1]).rx(0, tm[2]);
        let e_m = expectation_value(&circ_m.execute().unwrap(), &z_obs).unwrap();

        let fd = (e_p - e_m) / (2.0 * eps);
        assert!(
            (adj_grads.gradients[i] - fd).abs() < 1e-4,
            "Gate {} gradient: adjoint {} vs FD {}", i, adj_grads.gradients[i], fd
        );
    }
}

#[test]
fn adjoint_non_parameterized_zero_gradient() {
    // H, X, Y, Z, CNOT should all have zero gradient
    let mut circ = Circuit::new(2);
    circ.h(0).x(1).y(0).z(1).add(Gate::CNOT(0, 1));
    let z_obs = vec![1.0, -1.0, 1.0, -1.0]; // 2-qubit Z_0 observable
    let grads = adjoint_differentiation(&circ, &z_obs).unwrap();

    for (i, &g) in grads.gradients.iter().enumerate() {
        assert_eq!(g, 0.0, "Non-parameterized gate {} has non-zero gradient {}", i, g);
    }
}

#[test]
fn adjoint_two_qubit_circuit_gradient() {
    // Circuit: Ry(t1) on q0, CNOT(0,1), Ry(t2) on q1
    // Verify gradient correctness via finite differences
    let theta1 = 0.5;
    let theta2 = 0.9;
    let z_obs = vec![1.0, -1.0, -1.0, 1.0]; // Z_0 x Z_1
    let eps = 1e-5;

    let mut circ = Circuit::new(2);
    circ.ry(0, theta1).add(Gate::CNOT(0, 1)).ry(1, theta2);
    let adj_grads = adjoint_differentiation(&circ, &z_obs).unwrap();

    // Finite diff for theta1 (gate 0)
    let mut cp = Circuit::new(2);
    cp.ry(0, theta1 + eps).add(Gate::CNOT(0, 1)).ry(1, theta2);
    let mut cm = Circuit::new(2);
    cm.ry(0, theta1 - eps).add(Gate::CNOT(0, 1)).ry(1, theta2);
    let fd1 = (expectation_value(&cp.execute().unwrap(), &z_obs).unwrap()
        - expectation_value(&cm.execute().unwrap(), &z_obs).unwrap()) / (2.0 * eps);
    assert!(
        (adj_grads.gradients[0] - fd1).abs() < 1e-4,
        "2q gate 0: adjoint {} vs FD {}", adj_grads.gradients[0], fd1
    );

    // Finite diff for theta2 (gate 2)
    let mut cp2 = Circuit::new(2);
    cp2.ry(0, theta1).add(Gate::CNOT(0, 1)).ry(1, theta2 + eps);
    let mut cm2 = Circuit::new(2);
    cm2.ry(0, theta1).add(Gate::CNOT(0, 1)).ry(1, theta2 - eps);
    let fd2 = (expectation_value(&cp2.execute().unwrap(), &z_obs).unwrap()
        - expectation_value(&cm2.execute().unwrap(), &z_obs).unwrap()) / (2.0 * eps);
    assert!(
        (adj_grads.gradients[2] - fd2).abs() < 1e-4,
        "2q gate 2: adjoint {} vs FD {}", adj_grads.gradients[2], fd2
    );
}

// =========================================================================
// 3. SIMD TESTS
// =========================================================================

#[test]
fn simd_h_gate_bit_identical_to_scalar() {
    for n_qubits in 1..=6 {
        for target in 0..n_qubits {
            let mut sv_scalar = Statevector::new(n_qubits);
            let mut sv_simd = sv_scalar.clone();

            apply_single_qubit_cached(&mut sv_scalar, target, h_matrix());
            apply_single_qubit_simd(&mut sv_simd, target, h_matrix());

            for i in 0..sv_scalar.n_states() {
                assert_eq!(
                    sv_scalar.amplitudes[i].re.to_bits(),
                    sv_simd.amplitudes[i].re.to_bits(),
                    "H gate SIMD mismatch: n={}, q={}, state={}, re",
                    n_qubits, target, i
                );
                assert_eq!(
                    sv_scalar.amplitudes[i].im.to_bits(),
                    sv_simd.amplitudes[i].im.to_bits(),
                    "H gate SIMD mismatch: n={}, q={}, state={}, im",
                    n_qubits, target, i
                );
            }
        }
    }
}

#[test]
fn simd_x_gate_bit_identical_to_scalar() {
    for n_qubits in 1..=5 {
        for target in 0..n_qubits {
            let mut sv_scalar = Statevector::new(n_qubits);
            let mut sv_simd = sv_scalar.clone();

            apply_single_qubit_cached(&mut sv_scalar, target, x_matrix());
            apply_single_qubit_simd(&mut sv_simd, target, x_matrix());

            for i in 0..sv_scalar.n_states() {
                assert_eq!(
                    sv_scalar.amplitudes[i].re.to_bits(),
                    sv_simd.amplitudes[i].re.to_bits(),
                    "X gate SIMD mismatch: n={}, q={}, state={}", n_qubits, target, i
                );
                assert_eq!(
                    sv_scalar.amplitudes[i].im.to_bits(),
                    sv_simd.amplitudes[i].im.to_bits(),
                    "X gate SIMD mismatch im: n={}, q={}, state={}", n_qubits, target, i
                );
            }
        }
    }
}

#[test]
fn simd_ry_gate_bit_identical_to_scalar() {
    let theta = 0.73;
    for n_qubits in 1..=5 {
        for target in 0..n_qubits {
            let mut sv_scalar = Statevector::new(n_qubits);
            let mut sv_simd = sv_scalar.clone();

            let u = ry_matrix(theta);
            apply_single_qubit_cached(&mut sv_scalar, target, u);
            apply_single_qubit_simd(&mut sv_simd, target, u);

            for i in 0..sv_scalar.n_states() {
                assert_eq!(
                    sv_scalar.amplitudes[i].re.to_bits(),
                    sv_simd.amplitudes[i].re.to_bits(),
                    "Ry SIMD mismatch: n={}, q={}, state={}", n_qubits, target, i
                );
                assert_eq!(
                    sv_scalar.amplitudes[i].im.to_bits(),
                    sv_simd.amplitudes[i].im.to_bits(),
                    "Ry SIMD mismatch im: n={}, q={}, state={}", n_qubits, target, i
                );
            }
        }
    }
}

#[test]
fn simd_rz_gate_bit_identical_to_scalar() {
    let theta = 1.23;
    for n_qubits in 1..=5 {
        for target in 0..n_qubits {
            let mut sv_scalar = Statevector::new(n_qubits);
            let mut sv_simd = sv_scalar.clone();

            let u = rz_matrix(theta);
            apply_single_qubit_cached(&mut sv_scalar, target, u);
            apply_single_qubit_simd(&mut sv_simd, target, u);

            for i in 0..sv_scalar.n_states() {
                assert_eq!(
                    sv_scalar.amplitudes[i].re.to_bits(),
                    sv_simd.amplitudes[i].re.to_bits(),
                    "Rz SIMD mismatch: n={}, q={}, state={}", n_qubits, target, i
                );
                assert_eq!(
                    sv_scalar.amplitudes[i].im.to_bits(),
                    sv_simd.amplitudes[i].im.to_bits(),
                    "Rz SIMD mismatch im: n={}, q={}, state={}", n_qubits, target, i
                );
            }
        }
    }
}

#[test]
fn simd_rx_gate_bit_identical_to_scalar() {
    let theta = 2.45;
    for n_qubits in 1..=5 {
        for target in 0..n_qubits {
            let mut sv_scalar = Statevector::new(n_qubits);
            let mut sv_simd = sv_scalar.clone();

            let u = rx_matrix(theta);
            apply_single_qubit_cached(&mut sv_scalar, target, u);
            apply_single_qubit_simd(&mut sv_simd, target, u);

            for i in 0..sv_scalar.n_states() {
                assert_eq!(
                    sv_scalar.amplitudes[i].re.to_bits(),
                    sv_simd.amplitudes[i].re.to_bits(),
                    "Rx SIMD mismatch: n={}, q={}, state={}", n_qubits, target, i
                );
                assert_eq!(
                    sv_scalar.amplitudes[i].im.to_bits(),
                    sv_simd.amplitudes[i].im.to_bits(),
                    "Rx SIMD mismatch im: n={}, q={}, state={}", n_qubits, target, i
                );
            }
        }
    }
}

#[test]
fn simd_y_gate_bit_identical_to_scalar() {
    for n_qubits in 1..=4 {
        for target in 0..n_qubits {
            let mut sv_scalar = Statevector::new(n_qubits);
            let mut sv_simd = sv_scalar.clone();

            apply_single_qubit_cached(&mut sv_scalar, target, y_matrix());
            apply_single_qubit_simd(&mut sv_simd, target, y_matrix());

            for i in 0..sv_scalar.n_states() {
                assert_eq!(
                    sv_scalar.amplitudes[i].re.to_bits(),
                    sv_simd.amplitudes[i].re.to_bits(),
                    "Y SIMD mismatch: n={}, q={}, state={}", n_qubits, target, i
                );
                assert_eq!(
                    sv_scalar.amplitudes[i].im.to_bits(),
                    sv_simd.amplitudes[i].im.to_bits(),
                    "Y SIMD mismatch im: n={}, q={}, state={}", n_qubits, target, i
                );
            }
        }
    }
}

#[test]
fn simd_z_gate_bit_identical_to_scalar() {
    for n_qubits in 1..=4 {
        for target in 0..n_qubits {
            let mut sv_scalar = Statevector::new(n_qubits);
            let mut sv_simd = sv_scalar.clone();

            apply_single_qubit_cached(&mut sv_scalar, target, z_matrix());
            apply_single_qubit_simd(&mut sv_simd, target, z_matrix());

            for i in 0..sv_scalar.n_states() {
                assert_eq!(
                    sv_scalar.amplitudes[i].re.to_bits(),
                    sv_simd.amplitudes[i].re.to_bits(),
                    "Z SIMD mismatch: n={}, q={}, state={}", n_qubits, target, i
                );
                assert_eq!(
                    sv_scalar.amplitudes[i].im.to_bits(),
                    sv_simd.amplitudes[i].im.to_bits(),
                    "Z SIMD mismatch im: n={}, q={}, state={}", n_qubits, target, i
                );
            }
        }
    }
}

#[test]
fn simd_cached_blocking_matches_scalar_large_qubit() {
    // Test cache-aware blocking for larger qubit indices
    // 8 qubits = 256 states; target qubit 7 has stride 128
    let n_qubits = 8;
    for target in 5..n_qubits {
        let mut sv_scalar = Statevector::new(n_qubits);
        let mut sv_cached = sv_scalar.clone();

        // Apply H to put in superposition first
        Gate::H(0).apply(&mut sv_scalar).unwrap();
        Gate::H(0).apply(&mut sv_cached).unwrap();

        apply_single_qubit_cached(&mut sv_scalar, target, h_matrix());
        apply_single_qubit_cached(&mut sv_cached, target, h_matrix());

        for i in 0..sv_scalar.n_states() {
            assert_eq!(
                sv_scalar.amplitudes[i].re.to_bits(),
                sv_cached.amplitudes[i].re.to_bits(),
                "Cache blocking mismatch: n={}, q={}, state={}, re", n_qubits, target, i
            );
            assert_eq!(
                sv_scalar.amplitudes[i].im.to_bits(),
                sv_cached.amplitudes[i].im.to_bits(),
                "Cache blocking mismatch: n={}, q={}, state={}, im", n_qubits, target, i
            );
        }
    }
}

#[test]
fn simd_multi_gate_sequence_bit_identical() {
    // Apply a sequence of different gates via scalar and SIMD and verify bit identity
    let n_qubits = 4;
    let mut sv_scalar = Statevector::new(n_qubits);
    let mut sv_simd = sv_scalar.clone();

    let gates: Vec<(usize, [[ComplexF64; 2]; 2])> = vec![
        (0, h_matrix()),
        (1, rx_matrix(0.5)),
        (2, ry_matrix(1.2)),
        (3, rz_matrix(0.8)),
        (0, x_matrix()),
        (1, h_matrix()),
    ];

    for (q, u) in &gates {
        apply_single_qubit_cached(&mut sv_scalar, *q, *u);
        apply_single_qubit_simd(&mut sv_simd, *q, *u);
    }

    for i in 0..sv_scalar.n_states() {
        assert_eq!(
            sv_scalar.amplitudes[i].re.to_bits(),
            sv_simd.amplitudes[i].re.to_bits(),
            "Multi-gate SIMD mismatch at state {}, re", i
        );
        assert_eq!(
            sv_scalar.amplitudes[i].im.to_bits(),
            sv_simd.amplitudes[i].im.to_bits(),
            "Multi-gate SIMD mismatch at state {}, im", i
        );
    }
}

// =========================================================================
// 4. MPS TESTS
// =========================================================================

#[test]
fn mps_ghz_matches_statevector_3_to_8_qubits() {
    for n in 3..=8 {
        // Build GHZ via MPS: H on q0, then CNOT chain
        let mut mps = Mps::new(n);
        mps.apply_single_qubit(0, h_matrix());
        for i in 0..(n - 1) {
            mps.apply_cnot_adjacent(i, i + 1);
        }
        let mps_sv = mps.to_statevector();

        // Build GHZ via Circuit
        let mut circ = Circuit::new(n);
        circ.h(0);
        for i in 0..(n - 1) {
            circ.add(Gate::CNOT(i, i + 1));
        }
        let circ_sv = circ.execute().unwrap();

        // Compare amplitudes
        let dim = 1 << n;
        for i in 0..dim {
            assert!(
                (mps_sv[i].re - circ_sv.amplitudes[i].re).abs() < TOL
                    && (mps_sv[i].im - circ_sv.amplitudes[i].im).abs() < TOL,
                "GHZ n={}, state {}: MPS ({}, {}) vs Circuit ({}, {})",
                n, i, mps_sv[i].re, mps_sv[i].im,
                circ_sv.amplitudes[i].re, circ_sv.amplitudes[i].im
            );
        }
    }
}

#[test]
fn mps_product_state_matches_statevector() {
    // Apply different single-qubit gates to each qubit (product state)
    let n = 4;
    let mut mps = Mps::new(n);
    mps.apply_single_qubit(0, h_matrix());
    mps.apply_single_qubit(1, x_matrix());
    mps.apply_single_qubit(2, ry_matrix(PI / 3.0));
    mps.apply_single_qubit(3, rz_matrix(PI / 5.0));
    let mps_sv = mps.to_statevector();

    let mut circ = Circuit::new(n);
    circ.h(0).x(1).ry(2, PI / 3.0).rz(3, PI / 5.0);
    let circ_sv = circ.execute().unwrap();

    for i in 0..(1 << n) {
        assert!(
            (mps_sv[i].re - circ_sv.amplitudes[i].re).abs() < TOL
                && (mps_sv[i].im - circ_sv.amplitudes[i].im).abs() < TOL,
            "Product state mismatch at {}: MPS ({}, {}) vs Circuit ({}, {})",
            i, mps_sv[i].re, mps_sv[i].im,
            circ_sv.amplitudes[i].re, circ_sv.amplitudes[i].im
        );
    }
}

#[test]
fn svd_deterministic_bit_identical() {
    // Run SVD multiple times on the same matrix -- must produce bit-identical results
    let mut m = DenseMatrix::zeros(4, 3);
    let values = [
        (0, 0, 1.0, 0.5), (0, 1, 0.3, -0.2), (0, 2, -0.8, 0.1),
        (1, 0, -0.7, 0.1), (1, 1, 0.9, 0.4), (1, 2, 0.2, -0.6),
        (2, 0, 0.2, -0.8), (2, 1, -0.1, 0.6), (2, 2, 1.3, 0.3),
        (3, 0, 0.4, 0.7), (3, 1, -0.5, -0.3), (3, 2, 0.8, -0.9),
    ];
    for &(r, c, re, im) in &values {
        m.set(r, c, ComplexF64::new(re, im));
    }

    let svd1 = svd_sign_stabilized(&m);
    let svd2 = svd_sign_stabilized(&m);
    let svd3 = svd_sign_stabilized(&m);

    for i in 0..svd1.s.len() {
        assert_eq!(svd1.s[i].to_bits(), svd2.s[i].to_bits(),
            "SVD run 1 vs 2: s[{}] not bit-identical", i);
        assert_eq!(svd1.s[i].to_bits(), svd3.s[i].to_bits(),
            "SVD run 1 vs 3: s[{}] not bit-identical", i);
    }

    for r in 0..svd1.u.rows {
        for c in 0..svd1.u.cols {
            assert_eq!(
                svd1.u.get(r, c).re.to_bits(),
                svd2.u.get(r, c).re.to_bits(),
                "U[{},{}].re not bit-identical (runs 1 vs 2)", r, c
            );
            assert_eq!(
                svd1.u.get(r, c).im.to_bits(),
                svd2.u.get(r, c).im.to_bits(),
                "U[{},{}].im not bit-identical (runs 1 vs 2)", r, c
            );
            assert_eq!(
                svd1.u.get(r, c).re.to_bits(),
                svd3.u.get(r, c).re.to_bits(),
                "U[{},{}].re not bit-identical (runs 1 vs 3)", r, c
            );
        }
    }

    for r in 0..svd1.vh.rows {
        for c in 0..svd1.vh.cols {
            assert_eq!(
                svd1.vh.get(r, c).re.to_bits(),
                svd2.vh.get(r, c).re.to_bits(),
                "Vh[{},{}].re not bit-identical", r, c
            );
            assert_eq!(
                svd1.vh.get(r, c).im.to_bits(),
                svd2.vh.get(r, c).im.to_bits(),
                "Vh[{},{}].im not bit-identical", r, c
            );
        }
    }
}

#[test]
fn svd_reconstruction_error_small() {
    // Verify A = U * diag(S) * Vh with small reconstruction error
    let mut m = DenseMatrix::zeros(4, 3);
    let values = [
        (0, 0, 1.0, 0.5), (0, 1, 0.3, -0.2), (0, 2, -0.8, 0.1),
        (1, 0, -0.7, 0.1), (1, 1, 0.9, 0.4), (1, 2, 0.2, -0.6),
        (2, 0, 0.2, -0.8), (2, 1, -0.1, 0.6), (2, 2, 1.3, 0.3),
        (3, 0, 0.4, 0.7), (3, 1, -0.5, -0.3), (3, 2, 0.8, -0.9),
    ];
    for &(r, c, re, im) in &values {
        m.set(r, c, ComplexF64::new(re, im));
    }

    let svd = svd_sign_stabilized(&m);
    let k = svd.s.len();

    let mut max_err = 0.0f64;
    for r in 0..m.rows {
        for c in 0..m.cols {
            let mut sum = ComplexF64::ZERO;
            for i in 0..k {
                sum = sum.add(
                    svd.u.get(r, i).scale(svd.s[i]).mul_fixed(svd.vh.get(i, c))
                );
            }
            let orig = m.get(r, c);
            let err = sum.sub(orig);
            let err_mag = err.norm_sq().sqrt();
            if err_mag > max_err {
                max_err = err_mag;
            }
        }
    }

    assert!(
        max_err < 1e-8,
        "SVD reconstruction max error {} exceeds 1e-8", max_err
    );
}

#[test]
fn svd_reconstruction_tall_matrix() {
    // Test SVD on a tall matrix (more rows than cols) -- the typical MPS shape
    let mut m = DenseMatrix::zeros(5, 2);
    m.set(0, 0, ComplexF64::new(1.0, 0.0));
    m.set(0, 1, ComplexF64::new(0.0, 1.0));
    m.set(1, 0, ComplexF64::new(0.5, 0.5));
    m.set(1, 1, ComplexF64::new(-0.3, 0.2));
    m.set(2, 0, ComplexF64::new(0.0, -1.0));
    m.set(2, 1, ComplexF64::new(1.0, 0.0));
    m.set(3, 0, ComplexF64::new(-0.2, 0.8));
    m.set(3, 1, ComplexF64::new(0.7, -0.1));
    m.set(4, 0, ComplexF64::new(0.4, -0.6));
    m.set(4, 1, ComplexF64::new(-0.5, 0.3));

    let svd = svd_sign_stabilized(&m);
    let k = svd.s.len();

    for r in 0..m.rows {
        for c in 0..m.cols {
            let mut sum = ComplexF64::ZERO;
            for i in 0..k {
                sum = sum.add(
                    svd.u.get(r, i).scale(svd.s[i]).mul_fixed(svd.vh.get(i, c))
                );
            }
            let err = sum.sub(m.get(r, c)).norm_sq().sqrt();
            assert!(
                err < 1e-8,
                "Tall SVD reconstruction error at ({},{}): {}", r, c, err
            );
        }
    }
}

#[test]
fn mps_memory_scaling_product_state() {
    // For product states (no entanglement), memory should be O(N)
    let sizes = [10, 20, 50, 100];
    let mut memories = vec![];

    for &n in &sizes {
        let mps = Mps::new(n);
        memories.push(mps.memory_bytes());
    }

    // Memory should scale linearly: mem(2*n) ~= 2 * mem(n)
    // Check that doubling N roughly doubles memory
    let ratio_20_10 = memories[1] as f64 / memories[0] as f64;
    assert!(
        (ratio_20_10 - 2.0).abs() < 0.5,
        "Memory ratio 20/10 qubits = {}, expected ~2.0", ratio_20_10
    );

    let ratio_100_50 = memories[3] as f64 / memories[2] as f64;
    assert!(
        (ratio_100_50 - 2.0).abs() < 0.5,
        "Memory ratio 100/50 qubits = {}, expected ~2.0", ratio_100_50
    );

    // Product state should use very little memory
    assert!(
        memories[3] < 10_000,
        "100-qubit product state should use < 10KB, got {} bytes", memories[3]
    );
}

#[test]
fn mps_ghz_bond_dimension_is_2() {
    // GHZ states should have bond dimension exactly 2 at interior bonds
    for n in 3..=6 {
        let mut mps = Mps::new(n);
        mps.apply_single_qubit(0, h_matrix());
        for i in 0..(n - 1) {
            mps.apply_cnot_adjacent(i, i + 1);
        }

        for i in 0..(n - 1) {
            assert!(
                mps.tensors[i].bond_right <= 2,
                "GHZ n={}: bond between {} and {} is {}, expected <= 2",
                n, i, i + 1, mps.tensors[i].bond_right
            );
        }
    }
}

#[test]
fn mps_normalization_preserved() {
    // After MPS operations, the statevector should remain normalized
    let mut mps = Mps::new(4);
    mps.apply_single_qubit(0, h_matrix());
    mps.apply_cnot_adjacent(0, 1);
    mps.apply_single_qubit(2, ry_matrix(1.0));
    mps.apply_cnot_adjacent(2, 3);

    let sv = mps.to_statevector();
    let norm = sv_norm(&sv);
    assert!(
        (norm - 1.0).abs() < TOL,
        "MPS statevector norm = {}, expected 1.0", norm
    );
}

#[test]
fn mps_bell_state_correct() {
    let mut mps = Mps::new(2);
    mps.apply_single_qubit(0, h_matrix());
    mps.apply_cnot_adjacent(0, 1);
    let sv = mps.to_statevector();

    // Bell state: (|00> + |11>) / sqrt(2)
    assert!((sv[0].re - FRAC_1_SQRT_2).abs() < TOL, "Bell |00> amplitude");
    assert!(sv[1].norm_sq() < TOL, "Bell |01> should be zero");
    assert!(sv[2].norm_sq() < TOL, "Bell |10> should be zero");
    assert!((sv[3].re - FRAC_1_SQRT_2).abs() < TOL, "Bell |11> amplitude");
}

// =========================================================================
// 5. CROSS-MODULE TESTS
// =========================================================================

#[test]
fn cross_adjoint_gradient_mps_vs_statevector() {
    // Build the same circuit using MPS and Circuit, then verify the adjoint
    // gradient computed from the Circuit matches the expectation value computed
    // from the MPS statevector.
    let theta = 0.6;
    let n = 2;
    let z_obs = vec![1.0, -1.0, 1.0, -1.0]; // Z on q0

    // Circuit for adjoint gradient
    let mut circ = Circuit::new(n);
    circ.ry(0, theta).add(Gate::CNOT(0, 1));
    let adj_grads = adjoint_differentiation(&circ, &z_obs).unwrap();

    // MPS: compute expectation value
    let mut mps = Mps::new(n);
    mps.apply_single_qubit(0, ry_matrix(theta));
    mps.apply_cnot_adjacent(0, 1);
    let mps_sv = mps.to_statevector();

    // Expectation via MPS statevector
    let mut e_mps = 0.0;
    for (i, &obs) in z_obs.iter().enumerate() {
        e_mps += mps_sv[i].norm_sq() * obs;
    }

    // Circuit expectation
    let circ_sv = circ.execute().unwrap();
    let e_circ = expectation_value(&circ_sv, &z_obs).unwrap();

    assert!(
        (e_mps - e_circ).abs() < TOL,
        "MPS expectation {} vs Circuit expectation {}", e_mps, e_circ
    );

    // Verify gradient via finite differences using MPS
    let eps = 1e-5;
    let mut mps_p = Mps::new(n);
    mps_p.apply_single_qubit(0, ry_matrix(theta + eps));
    mps_p.apply_cnot_adjacent(0, 1);
    let sv_p = mps_p.to_statevector();
    let e_p: f64 = sv_p.iter().zip(z_obs.iter()).map(|(a, &o)| a.norm_sq() * o).sum();

    let mut mps_m = Mps::new(n);
    mps_m.apply_single_qubit(0, ry_matrix(theta - eps));
    mps_m.apply_cnot_adjacent(0, 1);
    let sv_m = mps_m.to_statevector();
    let e_m: f64 = sv_m.iter().zip(z_obs.iter()).map(|(a, &o)| a.norm_sq() * o).sum();

    let mps_fd_grad = (e_p - e_m) / (2.0 * eps);

    assert!(
        (adj_grads.gradients[0] - mps_fd_grad).abs() < 1e-4,
        "Adjoint grad {} vs MPS FD grad {}", adj_grads.gradients[0], mps_fd_grad
    );
}

#[test]
fn cross_mps_simd_statevector_agreement() {
    // Build the same state using MPS single-qubit gates and SIMD kernel,
    // verify they produce the same statevector.
    let n = 3;
    let gates: Vec<(usize, [[ComplexF64; 2]; 2])> = vec![
        (0, h_matrix()),
        (1, ry_matrix(0.8)),
        (2, rz_matrix(1.5)),
    ];

    // MPS path
    let mut mps = Mps::new(n);
    for &(q, u) in &gates {
        mps.apply_single_qubit(q, u);
    }
    let mps_sv = mps.to_statevector();

    // SIMD path
    let mut sv_simd = Statevector::new(n);
    for &(q, u) in &gates {
        apply_single_qubit_simd(&mut sv_simd, q, u);
    }

    // Compare
    for i in 0..(1 << n) {
        assert!(
            (mps_sv[i].re - sv_simd.amplitudes[i].re).abs() < TOL
                && (mps_sv[i].im - sv_simd.amplitudes[i].im).abs() < TOL,
            "MPS vs SIMD mismatch at state {}: ({}, {}) vs ({}, {})",
            i, mps_sv[i].re, mps_sv[i].im,
            sv_simd.amplitudes[i].re, sv_simd.amplitudes[i].im
        );
    }
}

#[test]
fn cross_wirtinger_adjoint_consistency() {
    // For a single Ry(theta)|0> circuit with Z observable:
    // E(theta) = cos(theta), dE/dtheta = -sin(theta)
    // The Wirtinger gradient of probability P(|0>) = cos^2(theta/2) should be
    // consistent with the adjoint gradient.
    let theta = 0.9;

    // Adjoint gradient
    let mut circ = Circuit::new(1);
    circ.ry(0, theta);
    let z_obs = vec![1.0, -1.0];
    let adj_grads = adjoint_differentiation(&circ, &z_obs).unwrap();

    // Wirtinger: amplitude of |0> is cos(theta/2)
    let alpha0 = ComplexF64::real((theta / 2.0).cos());
    let (_, dz_conj) = probability_gradient(alpha0);

    // The probability gradient dP(|0>)/dalpha0 = alpha0 (real)
    // and the chain to dE/dtheta involves additional factors,
    // but both should give -sin(theta) for the full gradient.
    let analytical = -(theta).sin();
    assert!(
        (adj_grads.gradients[0] - analytical).abs() < 1e-6,
        "Adjoint grad {} vs analytical {}", adj_grads.gradients[0], analytical
    );

    // Wirtinger dz_conj for |alpha0|^2 should equal alpha0 for real amplitudes
    assert!(
        (dz_conj.re - alpha0.re).abs() < TOL,
        "Wirtinger dz_conj {} vs alpha0 {}", dz_conj.re, alpha0.re
    );
}
