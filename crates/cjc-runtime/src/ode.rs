//! ODE / PDE Solver Infrastructure — Minimal Primitives
//!
//! These are stub primitives designed for future library integration (Bastion).
//! They provide the foundational stepping functions; full solver loops and
//! adaptive algorithms will be built as CJC library code on top of these.

use crate::tensor::Tensor;
use cjc_repro::kahan_sum_f64;

// ---------------------------------------------------------------------------
// ODE Stepping Primitives
// ---------------------------------------------------------------------------

/// Single step of Euler's method: y_{n+1} = y_n + h * f(t_n, y_n).
///
/// # Arguments
/// * `y` - Current state vector (1D tensor)
/// * `dydt` - Derivative vector at current time (1D tensor, same shape as y)
/// * `h` - Step size
///
/// # Returns
/// New state vector y_{n+1}
pub fn ode_step_euler(y: &Tensor, dydt: &Tensor, h: f64) -> Tensor {
    let y_data = y.to_vec();
    let dy_data = dydt.to_vec();
    assert_eq!(y_data.len(), dy_data.len(), "ode_step_euler: y and dydt must have same length");

    let result: Vec<f64> = y_data.iter().zip(dy_data.iter())
        .map(|(&yi, &dyi)| yi + h * dyi)
        .collect();
    Tensor::from_vec_unchecked(result, y.shape())
}

/// Single step of classical RK4: 4th-order Runge-Kutta.
///
/// Takes four derivative evaluations (k1, k2, k3, k4) and combines them
/// using the standard RK4 formula:
///   y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
///
/// The caller is responsible for evaluating k1..k4 at the appropriate
/// intermediate points. This keeps the stepping primitive pure and
/// independent of the RHS function.
pub fn ode_step_rk4(y: &Tensor, k1: &Tensor, k2: &Tensor, k3: &Tensor, k4: &Tensor, h: f64) -> Tensor {
    let y_data = y.to_vec();
    let k1_data = k1.to_vec();
    let k2_data = k2.to_vec();
    let k3_data = k3.to_vec();
    let k4_data = k4.to_vec();
    let n = y_data.len();
    assert_eq!(k1_data.len(), n);
    assert_eq!(k2_data.len(), n);
    assert_eq!(k3_data.len(), n);
    assert_eq!(k4_data.len(), n);

    let h6 = h / 6.0;
    let result: Vec<f64> = (0..n)
        .map(|i| {
            // Use Kahan summation for the weighted sum to maintain determinism
            let terms = [
                k1_data[i],
                2.0 * k2_data[i],
                2.0 * k3_data[i],
                k4_data[i],
            ];
            y_data[i] + h6 * kahan_sum_f64(&terms)
        })
        .collect();
    Tensor::from_vec_unchecked(result, y.shape())
}

// ---------------------------------------------------------------------------
// PDE Stepping Primitives
// ---------------------------------------------------------------------------

/// 1D finite-difference Laplacian: d^2u/dx^2 ≈ (u[i-1] - 2*u[i] + u[i+1]) / dx^2
///
/// Boundary condition: Dirichlet (u[0] and u[n-1] are held fixed, not updated).
///
/// # Arguments
/// * `u` - Current field values (1D tensor of length n)
/// * `dx` - Grid spacing
///
/// # Returns
/// Laplacian approximation (1D tensor, same shape; boundary elements are 0.0)
pub fn pde_laplacian_1d(u: &Tensor, dx: f64) -> Tensor {
    let data = u.to_vec();
    let n = data.len();
    let dx2_inv = 1.0 / (dx * dx);
    let mut lap = vec![0.0_f64; n];

    for i in 1..n - 1 {
        lap[i] = (data[i - 1] - 2.0 * data[i] + data[i + 1]) * dx2_inv;
    }

    Tensor::from_vec_unchecked(lap, u.shape())
}

/// Single explicit Euler step for a heat/diffusion PDE:
///   u_{n+1} = u_n + dt * alpha * laplacian(u_n)
///
/// # Arguments
/// * `u` - Current field values
/// * `alpha` - Diffusion coefficient
/// * `dt` - Time step
/// * `dx` - Spatial grid spacing
///
/// # Returns
/// Updated field values
pub fn pde_step_diffusion(u: &Tensor, alpha: f64, dt: f64, dx: f64) -> Tensor {
    let lap = pde_laplacian_1d(u, dx);
    let u_data = u.to_vec();
    let lap_data = lap.to_vec();
    let result: Vec<f64> = u_data.iter().zip(lap_data.iter())
        .map(|(&ui, &li)| ui + dt * alpha * li)
        .collect();
    Tensor::from_vec_unchecked(result, u.shape())
}

// ---------------------------------------------------------------------------
// Symbolic Differentiation Primitives
// ---------------------------------------------------------------------------

/// Symbolic expression representation for automatic symbolic differentiation.
///
/// These are value-level symbolic expressions that can be differentiated
/// symbolically before evaluation. This provides exact derivatives without
/// numerical error.
#[derive(Debug, Clone, PartialEq)]
pub enum SymExpr {
    /// Constant value
    Const(f64),
    /// Variable reference (by name)
    Var(String),
    /// Addition
    Add(Box<SymExpr>, Box<SymExpr>),
    /// Multiplication
    Mul(Box<SymExpr>, Box<SymExpr>),
    /// Power: base^exponent (exponent is a constant)
    Pow(Box<SymExpr>, f64),
    /// Sine
    Sin(Box<SymExpr>),
    /// Cosine
    Cos(Box<SymExpr>),
    /// Exponential
    Exp(Box<SymExpr>),
    /// Natural logarithm
    Ln(Box<SymExpr>),
    /// Negation
    Neg(Box<SymExpr>),
}

impl SymExpr {
    /// Symbolically differentiate with respect to variable `var`.
    pub fn differentiate(&self, var: &str) -> SymExpr {
        match self {
            SymExpr::Const(_) => SymExpr::Const(0.0),
            SymExpr::Var(name) => {
                if name == var {
                    SymExpr::Const(1.0)
                } else {
                    SymExpr::Const(0.0)
                }
            }
            SymExpr::Add(a, b) => SymExpr::Add(
                Box::new(a.differentiate(var)),
                Box::new(b.differentiate(var)),
            ),
            SymExpr::Mul(a, b) => {
                // Product rule: (f*g)' = f'*g + f*g'
                SymExpr::Add(
                    Box::new(SymExpr::Mul(
                        Box::new(a.differentiate(var)),
                        b.clone(),
                    )),
                    Box::new(SymExpr::Mul(
                        a.clone(),
                        Box::new(b.differentiate(var)),
                    )),
                )
            }
            SymExpr::Pow(base, exp) => {
                // d/dx [f^n] = n * f^(n-1) * f'
                SymExpr::Mul(
                    Box::new(SymExpr::Mul(
                        Box::new(SymExpr::Const(*exp)),
                        Box::new(SymExpr::Pow(base.clone(), exp - 1.0)),
                    )),
                    Box::new(base.differentiate(var)),
                )
            }
            SymExpr::Sin(inner) => {
                // d/dx sin(f) = cos(f) * f'
                SymExpr::Mul(
                    Box::new(SymExpr::Cos(inner.clone())),
                    Box::new(inner.differentiate(var)),
                )
            }
            SymExpr::Cos(inner) => {
                // d/dx cos(f) = -sin(f) * f'
                SymExpr::Mul(
                    Box::new(SymExpr::Neg(Box::new(SymExpr::Sin(inner.clone())))),
                    Box::new(inner.differentiate(var)),
                )
            }
            SymExpr::Exp(inner) => {
                // d/dx exp(f) = exp(f) * f'
                SymExpr::Mul(
                    Box::new(SymExpr::Exp(inner.clone())),
                    Box::new(inner.differentiate(var)),
                )
            }
            SymExpr::Ln(inner) => {
                // d/dx ln(f) = f' / f
                SymExpr::Mul(
                    Box::new(SymExpr::Pow(inner.clone(), -1.0)),
                    Box::new(inner.differentiate(var)),
                )
            }
            SymExpr::Neg(inner) => {
                SymExpr::Neg(Box::new(inner.differentiate(var)))
            }
        }
    }

    /// Evaluate the symbolic expression with the given variable bindings.
    pub fn eval(&self, bindings: &std::collections::BTreeMap<String, f64>) -> f64 {
        match self {
            SymExpr::Const(c) => *c,
            SymExpr::Var(name) => *bindings.get(name).unwrap_or(&0.0),
            SymExpr::Add(a, b) => a.eval(bindings) + b.eval(bindings),
            SymExpr::Mul(a, b) => a.eval(bindings) * b.eval(bindings),
            SymExpr::Pow(base, exp) => base.eval(bindings).powf(*exp),
            SymExpr::Sin(inner) => inner.eval(bindings).sin(),
            SymExpr::Cos(inner) => inner.eval(bindings).cos(),
            SymExpr::Exp(inner) => inner.eval(bindings).exp(),
            SymExpr::Ln(inner) => inner.eval(bindings).ln(),
            SymExpr::Neg(inner) => -inner.eval(bindings),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_euler_step() {
        let y = Tensor::from_vec_unchecked(vec![1.0, 0.0], &[2]);
        let dydt = Tensor::from_vec_unchecked(vec![0.0, 1.0], &[2]);
        let y1 = ode_step_euler(&y, &dydt, 0.1);
        let result = y1.to_vec();
        assert!((result[0] - 1.0).abs() < 1e-15);
        assert!((result[1] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn test_rk4_step_constant() {
        let y = Tensor::from_vec_unchecked(vec![1.0], &[1]);
        let k = Tensor::from_vec_unchecked(vec![2.0], &[1]);
        let y1 = ode_step_rk4(&y, &k, &k, &k, &k, 0.1);
        // y1 = 1.0 + (0.1/6)*(2+4+4+2) = 1.0 + 0.2 = 1.2
        assert!((y1.to_vec()[0] - 1.2).abs() < 1e-14);
    }

    #[test]
    fn test_laplacian_1d() {
        // u = [0, 1, 4, 9, 16] (x^2 at x=0..4, dx=1)
        // d^2u/dx^2 = 2 everywhere interior
        let u = Tensor::from_vec_unchecked(vec![0.0, 1.0, 4.0, 9.0, 16.0], &[5]);
        let lap = pde_laplacian_1d(&u, 1.0);
        let data = lap.to_vec();
        assert!((data[0] - 0.0).abs() < 1e-14); // boundary
        assert!((data[1] - 2.0).abs() < 1e-14);
        assert!((data[2] - 2.0).abs() < 1e-14);
        assert!((data[3] - 2.0).abs() < 1e-14);
        assert!((data[4] - 0.0).abs() < 1e-14); // boundary
    }

    #[test]
    fn test_symbolic_diff_polynomial() {
        // f(x) = x^3, f'(x) = 3*x^2
        let expr = SymExpr::Pow(Box::new(SymExpr::Var("x".into())), 3.0);
        let deriv = expr.differentiate("x");

        let mut bindings = BTreeMap::new();
        bindings.insert("x".into(), 2.0);

        let val = deriv.eval(&bindings);
        assert!((val - 12.0).abs() < 1e-12); // 3 * 2^2 = 12
    }

    #[test]
    fn test_symbolic_diff_sin() {
        // f(x) = sin(x), f'(x) = cos(x)
        let expr = SymExpr::Sin(Box::new(SymExpr::Var("x".into())));
        let deriv = expr.differentiate("x");

        let mut bindings = BTreeMap::new();
        bindings.insert("x".into(), 0.0);

        let val = deriv.eval(&bindings);
        assert!((val - 1.0).abs() < 1e-12); // cos(0) = 1
    }
}
