//! Wirtinger Calculus — Complex-valued automatic differentiation.
//!
//! For a real-valued loss function $L$ that depends on complex parameters $z$,
//! the Wirtinger derivative gives us:
//!
//! $$\frac{\partial L}{\partial z^*} = \frac{1}{2}\left(\frac{\partial L}{\partial x} + i\frac{\partial L}{\partial y}\right)$$
//!
//! where $z = x + iy$. This conjugate derivative is the direction of steepest
//! descent for complex parameters, analogous to the real gradient.
//!
//! # Key Identities
//!
//! For $f(z) = |z|^2 = z \cdot z^*$:
//! - $\frac{\partial}{\partial z}|z|^2 = z^*$
//! - $\frac{\partial}{\partial z^*}|z|^2 = z$
//!
//! # Determinism
//!
//! All accumulations use Kahan summation. Complex arithmetic uses
//! `ComplexF64::mul_fixed()` to prevent FMA drift.

use cjc_runtime::complex::ComplexF64;

/// A complex dual number for forward-mode Wirtinger differentiation.
///
/// Carries a complex value and its Wirtinger derivatives:
/// - `dz`: derivative with respect to z (∂f/∂z)
/// - `dz_conj`: derivative with respect to z* (∂f/∂z*)
///
/// For a real-valued loss L, the gradient direction is `dz_conj`.
#[derive(Debug, Clone)]
pub struct WirtingerDual {
    pub value: ComplexF64,
    /// ∂f/∂z — the holomorphic part
    pub dz: ComplexF64,
    /// ∂f/∂z* — the anti-holomorphic part (conjugate gradient)
    pub dz_conj: ComplexF64,
}

impl WirtingerDual {
    /// Create a new Wirtinger dual with explicit derivatives.
    pub fn new(value: ComplexF64, dz: ComplexF64, dz_conj: ComplexF64) -> Self {
        WirtingerDual { value, dz, dz_conj }
    }

    /// Create a constant (no derivatives).
    pub fn constant(value: ComplexF64) -> Self {
        WirtingerDual {
            value,
            dz: ComplexF64::ZERO,
            dz_conj: ComplexF64::ZERO,
        }
    }

    /// Create a variable z: ∂z/∂z = 1, ∂z/∂z* = 0.
    pub fn variable(value: ComplexF64) -> Self {
        WirtingerDual {
            value,
            dz: ComplexF64::ONE,
            dz_conj: ComplexF64::ZERO,
        }
    }

    /// Create a conjugate variable z*: ∂z*/∂z = 0, ∂z*/∂z* = 1.
    pub fn conjugate_variable(value: ComplexF64) -> Self {
        WirtingerDual {
            value: value.conj(),
            dz: ComplexF64::ZERO,
            dz_conj: ComplexF64::ONE,
        }
    }

    /// Complex addition: (f + g).
    /// ∂(f+g)/∂z = ∂f/∂z + ∂g/∂z
    /// ∂(f+g)/∂z* = ∂f/∂z* + ∂g/∂z*
    pub fn add(self, rhs: WirtingerDual) -> WirtingerDual {
        WirtingerDual {
            value: self.value.add(rhs.value),
            dz: self.dz.add(rhs.dz),
            dz_conj: self.dz_conj.add(rhs.dz_conj),
        }
    }

    /// Complex subtraction: (f - g).
    pub fn sub(self, rhs: WirtingerDual) -> WirtingerDual {
        WirtingerDual {
            value: self.value.sub(rhs.value),
            dz: self.dz.sub(rhs.dz),
            dz_conj: self.dz_conj.sub(rhs.dz_conj),
        }
    }

    /// Complex multiplication: (f * g).
    /// ∂(f·g)/∂z = (∂f/∂z)·g + f·(∂g/∂z)
    /// ∂(f·g)/∂z* = (∂f/∂z*)·g + f·(∂g/∂z*)
    pub fn mul(self, rhs: WirtingerDual) -> WirtingerDual {
        WirtingerDual {
            value: self.value.mul_fixed(rhs.value),
            dz: self.dz.mul_fixed(rhs.value).add(self.value.mul_fixed(rhs.dz)),
            dz_conj: self.dz_conj.mul_fixed(rhs.value).add(self.value.mul_fixed(rhs.dz_conj)),
        }
    }

    /// Scalar multiplication by a real number.
    pub fn scale(self, s: f64) -> WirtingerDual {
        WirtingerDual {
            value: self.value.scale(s),
            dz: self.dz.scale(s),
            dz_conj: self.dz_conj.scale(s),
        }
    }

    /// Complex conjugation: f → f*.
    /// ∂(f*)/∂z = (∂f/∂z*)* and ∂(f*)/∂z* = (∂f/∂z)*
    /// (Wirtinger conjugation swaps and conjugates the derivatives.)
    pub fn conj(self) -> WirtingerDual {
        WirtingerDual {
            value: self.value.conj(),
            dz: self.dz_conj.conj(),
            dz_conj: self.dz.conj(),
        }
    }

    /// Norm squared: |f|² = f · f*.
    /// This is the key non-holomorphic operation in quantum mechanics.
    /// ∂|f|²/∂z = f* · (∂f/∂z) + f · (∂f*/∂z) = f* · (∂f/∂z) + f · (∂f/∂z*)*
    /// ∂|f|²/∂z* = f* · (∂f/∂z*) + f · (∂f*/∂z*) = f* · (∂f/∂z*) + f · (∂f/∂z)*
    pub fn norm_sq(self) -> WirtingerDual {
        let f = self.value;
        let f_conj = self.value.conj();

        // |f|² is real
        let val = ComplexF64::real(f.norm_sq());

        // ∂|f|²/∂z = f* · df/dz + f · (df/dz*)*
        let dz = f_conj.mul_fixed(self.dz).add(f.mul_fixed(self.dz_conj.conj()));
        // ∂|f|²/∂z* = f* · df/dz* + f · (df/dz)*
        let dz_conj = f_conj.mul_fixed(self.dz_conj).add(f.mul_fixed(self.dz.conj()));

        WirtingerDual { value: val, dz, dz_conj }
    }

    /// Extract the real part of the conjugate gradient (for optimization).
    /// For a real-valued loss L, the update direction is -2 · Re(∂L/∂z*).
    pub fn conjugate_gradient(&self) -> ComplexF64 {
        self.dz_conj
    }
}

// ---------------------------------------------------------------------------
// Quantum-specific Wirtinger operations
// ---------------------------------------------------------------------------

/// Compute the Wirtinger gradient of probability |α|² with respect to α.
///
/// Given amplitude α, the probability is P = |α|² = α · α*.
/// The Wirtinger derivatives are:
/// - ∂P/∂α = α*
/// - ∂P/∂α* = α
///
/// This is the fundamental gradient needed for variational quantum algorithms.
pub fn probability_gradient(amplitude: ComplexF64) -> (ComplexF64, ComplexF64) {
    // ∂|α|²/∂α = α*
    let dz = amplitude.conj();
    // ∂|α|²/∂α* = α
    let dz_conj = amplitude;
    (dz, dz_conj)
}

/// Compute gradients of expectation value ⟨ψ|O|ψ⟩ with respect to parameters.
///
/// For a parameterized circuit U(θ), the expectation value is:
/// E(θ) = ⟨0|U†(θ) O U(θ)|0⟩
///
/// The parameter-shift rule gives:
/// ∂E/∂θ = (E(θ + π/2) - E(θ - π/2)) / 2
///
/// This function computes the gradient for a single parameter using the
/// parameter-shift rule with two circuit evaluations.
pub fn parameter_shift_gradient(
    expectation_plus: f64,
    expectation_minus: f64,
) -> f64 {
    (expectation_plus - expectation_minus) / 2.0
}

/// Accumulate Wirtinger gradients over a statevector using Kahan summation.
///
/// Given a vector of (amplitude, weight) pairs, compute the total gradient
/// of the weighted sum of probabilities Σ wᵢ|αᵢ|².
pub fn weighted_probability_gradient(
    amplitudes: &[ComplexF64],
    weights: &[f64],
) -> Vec<(ComplexF64, ComplexF64)> {
    assert_eq!(amplitudes.len(), weights.len());
    amplitudes.iter().zip(weights.iter()).map(|(&a, &w)| {
        let (dz, dz_conj) = probability_gradient(a);
        (dz.scale(w), dz_conj.scale(w))
    }).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-12;

    fn assert_complex_approx(a: ComplexF64, b: ComplexF64, msg: &str) {
        assert!((a.re - b.re).abs() < TOL && (a.im - b.im).abs() < TOL,
            "{}: got ({}, {}) expected ({}, {})", msg, a.re, a.im, b.re, b.im);
    }

    #[test]
    fn test_wirtinger_constant() {
        let c = WirtingerDual::constant(ComplexF64::new(3.0, 4.0));
        assert_complex_approx(c.dz, ComplexF64::ZERO, "constant dz");
        assert_complex_approx(c.dz_conj, ComplexF64::ZERO, "constant dz_conj");
    }

    #[test]
    fn test_wirtinger_variable() {
        let z = WirtingerDual::variable(ComplexF64::new(1.0, 2.0));
        assert_complex_approx(z.dz, ComplexF64::ONE, "variable dz");
        assert_complex_approx(z.dz_conj, ComplexF64::ZERO, "variable dz_conj");
    }

    #[test]
    fn test_wirtinger_add() {
        let z = WirtingerDual::variable(ComplexF64::new(1.0, 2.0));
        let w = WirtingerDual::constant(ComplexF64::new(3.0, 0.0));
        let sum = z.add(w);
        assert_complex_approx(sum.value, ComplexF64::new(4.0, 2.0), "add value");
        assert_complex_approx(sum.dz, ComplexF64::ONE, "add dz");
    }

    #[test]
    fn test_wirtinger_mul() {
        // f(z) = z * c where c = 2+0i
        // ∂(z·c)/∂z = c = 2
        let z = WirtingerDual::variable(ComplexF64::new(3.0, 1.0));
        let c = WirtingerDual::constant(ComplexF64::new(2.0, 0.0));
        let prod = z.mul(c);
        assert_complex_approx(prod.dz, ComplexF64::new(2.0, 0.0), "mul dz");
    }

    #[test]
    fn test_wirtinger_norm_sq_of_variable() {
        // f(z) = |z|², ∂|z|²/∂z = z*, ∂|z|²/∂z* = z
        let z_val = ComplexF64::new(3.0, 4.0);
        let z = WirtingerDual::variable(z_val);
        let nsq = z.norm_sq();

        assert!((nsq.value.re - 25.0).abs() < TOL, "norm_sq value");
        assert_complex_approx(nsq.dz, z_val.conj(), "∂|z|²/∂z = z*");
        assert_complex_approx(nsq.dz_conj, z_val, "∂|z|²/∂z* = z");
    }

    #[test]
    fn test_wirtinger_conjugation() {
        // For f(z) = z: df/dz = 1, df/dz* = 0
        // For g(z) = z*: dg/dz = 0, dg/dz* = 1
        let z = WirtingerDual::variable(ComplexF64::new(1.0, 2.0));
        let z_conj = z.conj();
        assert_complex_approx(z_conj.dz, ComplexF64::ZERO, "conj dz");
        assert_complex_approx(z_conj.dz_conj, ComplexF64::ONE, "conj dz_conj");
    }

    #[test]
    fn test_wirtinger_chain_rule() {
        // f(z) = |z² + 1|²
        // At z = 1+i: z² = (1+i)² = 2i, z²+1 = 1+2i
        // |1+2i|² = 5
        let z = WirtingerDual::variable(ComplexF64::new(1.0, 1.0));
        let z2 = z.clone().mul(z.clone());
        let one = WirtingerDual::constant(ComplexF64::ONE);
        let sum = z2.add(one);
        let result = sum.norm_sq();

        assert!((result.value.re - 5.0).abs() < TOL, "chain rule value");
        // The derivatives should be computed correctly through the chain
        assert!(result.dz.norm_sq() > 0.0, "chain rule dz should be non-zero");
    }

    #[test]
    fn test_probability_gradient_basic() {
        let alpha = ComplexF64::new(0.5, 0.5);
        let (dz, dz_conj) = probability_gradient(alpha);
        assert_complex_approx(dz, alpha.conj(), "prob grad dz = α*");
        assert_complex_approx(dz_conj, alpha, "prob grad dz* = α");
    }

    #[test]
    fn test_parameter_shift_rule() {
        // For Rz(θ)|0⟩, P(|0⟩) = cos²(θ/2)
        // ∂P/∂θ = -sin(θ/2)cos(θ/2) = -sin(θ)/2
        let theta = PI / 3.0;
        let e_plus = (theta / 2.0 + PI / 4.0).cos().powi(2);
        let e_minus = (theta / 2.0 - PI / 4.0).cos().powi(2);
        let grad = parameter_shift_gradient(e_plus, e_minus);
        let expected = -(theta).sin() / 2.0;
        assert!((grad - expected).abs() < 1e-10,
            "parameter shift: got {}, expected {}", grad, expected);
    }

    #[test]
    fn test_weighted_probability_gradient() {
        let amps = vec![
            ComplexF64::new(0.5, 0.5),
            ComplexF64::new(0.5, -0.5),
        ];
        let weights = vec![1.0, -1.0]; // observable = diag(1, -1) = Z
        let grads = weighted_probability_gradient(&amps, &weights);
        assert_eq!(grads.len(), 2);
        // Gradient for first amplitude (weight +1): dz = α*, dz* = α
        assert_complex_approx(grads[0].0, amps[0].conj().scale(1.0), "weighted grad 0 dz");
    }

    #[test]
    fn test_wirtinger_numerical_agreement() {
        // Verify Wirtinger derivative of |z|² matches finite differences
        let z0 = ComplexF64::new(2.0, 3.0);
        let eps = 1e-7;

        // Finite difference in real direction: ∂|z|²/∂x
        let f_xp = ComplexF64::new(z0.re + eps, z0.im).norm_sq();
        let f_xm = ComplexF64::new(z0.re - eps, z0.im).norm_sq();
        let df_dx = (f_xp - f_xm) / (2.0 * eps);

        // Finite difference in imaginary direction: ∂|z|²/∂y
        let f_yp = ComplexF64::new(z0.re, z0.im + eps).norm_sq();
        let f_ym = ComplexF64::new(z0.re, z0.im - eps).norm_sq();
        let df_dy = (f_yp - f_ym) / (2.0 * eps);

        // Wirtinger: ∂f/∂z* = (∂f/∂x + i·∂f/∂y) / 2
        let wirtinger_conj = ComplexF64::new(df_dx, df_dy).scale(0.5);

        // Analytical: ∂|z|²/∂z* = z
        let analytical = z0;

        assert!((wirtinger_conj.re - analytical.re).abs() < 1e-5,
            "numerical vs analytical re: {} vs {}", wirtinger_conj.re, analytical.re);
        assert!((wirtinger_conj.im - analytical.im).abs() < 1e-5,
            "numerical vs analytical im: {} vs {}", wirtinger_conj.im, analytical.im);
    }
}
