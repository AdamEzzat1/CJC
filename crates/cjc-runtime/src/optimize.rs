//! Optimization & Root Finding — deterministic numerical solvers.
//!
//! # Determinism Contract
//!
//! All algorithms in this module are fully deterministic: given the same inputs
//! and the same objective/gradient functions, they produce **bit-identical** results
//! across runs. Floating-point reductions use `binned_sum_f64` from the accumulator
//! module to avoid ordering-dependent rounding.
//!
//! # Scalar Root Finding
//!
//! - [`bisect`] — bisection method (guaranteed convergence for bracketed roots)
//! - [`brentq`] — Brent's method (IQI + bisection fallback, superlinear convergence)
//! - [`newton_scalar`] — Newton-Raphson (quadratic convergence near root)
//! - [`secant`] — secant method (superlinear convergence, no derivative needed)
//!
//! # Unconstrained Optimization
//!
//! - [`minimize_gd`] — gradient descent with fixed learning rate
//! - [`minimize_bfgs`] — BFGS quasi-Newton with Armijo line search
//! - [`minimize_lbfgs`] — limited-memory BFGS with m history vectors
//! - [`minimize_nelder_mead`] — Nelder-Mead simplex (derivative-free)

use crate::accumulator::binned_sum_f64;

// ---------------------------------------------------------------------------
// Result type for vector optimizers
// ---------------------------------------------------------------------------

/// Result of an unconstrained optimization run.
#[derive(Debug, Clone)]
pub struct OptResult {
    /// Optimal point found.
    pub x: Vec<f64>,
    /// Objective function value at `x`.
    pub fun: f64,
    /// Number of iterations performed.
    pub niter: usize,
    /// Whether the solver met the requested tolerance.
    pub converged: bool,
}

// ===========================================================================
// Scalar Root Finding
// ===========================================================================

/// Bisection method for scalar root finding.
///
/// Finds `x` in `[a, b]` such that `|f(x)| < tol`, given that `f(a)` and
/// `f(b)` have opposite signs. Returns `Err` if the bracket is invalid.
pub fn bisect(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut lo = a;
    let mut hi = b;
    let mut f_lo = f(lo);
    let f_hi = f(hi);

    if f_lo * f_hi > 0.0 {
        return Err(format!(
            "bisect: f(a) and f(b) must have opposite signs, got f({})={}, f({})={}",
            a, f_lo, b, f_hi
        ));
    }

    // If one endpoint is already a root, return it immediately.
    if f_lo == 0.0 {
        return Ok(lo);
    }
    if f_hi == 0.0 {
        return Ok(hi);
    }

    for _ in 0..max_iter {
        let mid = lo + (hi - lo) * 0.5; // avoids overflow vs (lo+hi)/2
        let f_mid = f(mid);

        if f_mid.abs() < tol || (hi - lo) * 0.5 < tol {
            return Ok(mid);
        }

        if f_lo * f_mid < 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }

    // Return best estimate even if not fully converged.
    Ok(lo + (hi - lo) * 0.5)
}

/// Brent's method for scalar root finding.
///
/// Combines inverse quadratic interpolation (IQI) with bisection fallback for
/// robust, superlinear convergence. Requires `f(a)` and `f(b)` to have opposite
/// signs.
pub fn brentq(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);

    if fa * fb > 0.0 {
        return Err(format!(
            "brentq: f(a) and f(b) must have opposite signs, got f({})={}, f({})={}",
            a, fa, b, fb
        ));
    }

    if fa.abs() < fb.abs() {
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0_f64; // previous step size (only used when mflag == false)

    for _ in 0..max_iter {
        if fb.abs() < tol {
            return Ok(b);
        }
        if fa.abs() < tol {
            return Ok(a);
        }
        if (b - a).abs() < tol {
            return Ok(b);
        }

        // Attempt inverse quadratic interpolation or secant.
        let s = if (fa - fc).abs() > f64::EPSILON && (fb - fc).abs() > f64::EPSILON {
            // Inverse quadratic interpolation.
            let t1 = a * fb * fc / ((fa - fb) * (fa - fc));
            let t2 = b * fa * fc / ((fb - fa) * (fb - fc));
            let t3 = c * fa * fb / ((fc - fa) * (fc - fb));
            binned_sum_f64(&[t1, t2, t3])
        } else {
            // Secant method.
            b - fb * (b - a) / (fb - fa)
        };

        // Conditions for rejecting `s` in favour of bisection.
        let mid = (a + b) * 0.5;
        let cond1 = {
            // s not between (3a+b)/4 and b
            let lo = if (3.0 * a + b) / 4.0 < b {
                (3.0 * a + b) / 4.0
            } else {
                b
            };
            let hi = if (3.0 * a + b) / 4.0 > b {
                (3.0 * a + b) / 4.0
            } else {
                b
            };
            s < lo || s > hi
        };
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() * 0.5;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() * 0.5;
        let cond4 = mflag && (b - c).abs() < tol;
        let cond5 = !mflag && (c - d).abs() < tol;

        let s = if cond1 || cond2 || cond3 || cond4 || cond5 {
            mflag = true;
            mid
        } else {
            mflag = false;
            s
        };

        let fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Keep |f(a)| >= |f(b)| so b is the better approximation.
        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }
    }

    Ok(b)
}

/// Newton-Raphson method for scalar root finding.
///
/// Uses `f` and its derivative `df` to iterate `x_{k+1} = x_k - f(x_k)/df(x_k)`.
/// Returns `Err` if the derivative is zero at any iterate.
pub fn newton_scalar(
    f: &dyn Fn(f64) -> f64,
    df: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut x = x0;

    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol {
            return Ok(x);
        }

        let dfx = df(x);
        if dfx.abs() < f64::EPSILON {
            return Err(format!(
                "newton_scalar: derivative is zero at x={}, cannot continue",
                x
            ));
        }

        x = x - fx / dfx;
    }

    if f(x).abs() < tol {
        Ok(x)
    } else {
        Err(format!(
            "newton_scalar: did not converge after {} iterations, x={}, f(x)={}",
            max_iter,
            x,
            f(x)
        ))
    }
}

/// Secant method for scalar root finding.
///
/// A derivative-free variant of Newton's method using finite difference
/// approximation of the derivative from the two most recent iterates.
pub fn secant(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    x1: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut xp = x0; // x_{k-1}
    let mut xc = x1; // x_k
    let mut fp = f(xp);
    let mut fc = f(xc);

    for _ in 0..max_iter {
        if fc.abs() < tol {
            return Ok(xc);
        }

        let denom = fc - fp;
        if denom.abs() < f64::EPSILON {
            return Err(format!(
                "secant: division by zero (f(x0)={}, f(x1)={} are too close)",
                fp, fc
            ));
        }

        let xn = xc - fc * (xc - xp) / denom;
        xp = xc;
        fp = fc;
        xc = xn;
        fc = f(xc);
    }

    if fc.abs() < tol {
        Ok(xc)
    } else {
        Err(format!(
            "secant: did not converge after {} iterations, x={}, f(x)={}",
            max_iter, xc, fc
        ))
    }
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/// Deterministic L2 norm of a vector using binned summation.
fn norm_l2(v: &[f64]) -> f64 {
    let sq: Vec<f64> = v.iter().map(|&x| x * x).collect();
    binned_sum_f64(&sq).sqrt()
}

/// Deterministic dot product using binned summation.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let prods: Vec<f64> = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).collect();
    binned_sum_f64(&prods)
}

/// Armijo (backtracking) line search.
///
/// Finds a step size `alpha` satisfying the Armijo sufficient decrease condition:
///   `f(x + alpha * d) <= f(x) + c * alpha * grad^T d`
///
/// Parameters:
/// - `f`: objective function
/// - `x`: current point
/// - `d`: search direction
/// - `grad`: gradient at `x`
/// - `alpha0`: initial step size
/// - `c`: sufficient decrease parameter (typically 1e-4)
/// - `rho`: backtracking factor (typically 0.5)
fn armijo_line_search(
    f: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    d: &[f64],
    grad: &[f64],
    alpha0: f64,
    c: f64,
    rho: f64,
) -> f64 {
    let n = x.len();
    let f0 = f(x);
    let slope = dot(grad, d); // directional derivative

    let mut alpha = alpha0;
    let mut x_new = vec![0.0; n];

    // Cap iterations to prevent infinite loops on pathological functions.
    for _ in 0..60 {
        for i in 0..n {
            x_new[i] = x[i] + alpha * d[i];
        }
        let f_new = f(&x_new);
        if f_new <= f0 + c * alpha * slope {
            return alpha;
        }
        alpha *= rho;
    }

    alpha
}

// ===========================================================================
// Unconstrained Optimization — Vector
// ===========================================================================

/// Gradient descent with fixed learning rate.
///
/// Minimizes `f` starting from `x0` by iterating `x_{k+1} = x_k - lr * grad(x_k)`.
pub fn minimize_gd(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    lr: f64,
    max_iter: usize,
    tol: f64,
) -> OptResult {
    let n = x0.len();
    let mut x = x0.to_vec();

    for iter in 0..max_iter {
        let g = grad(&x);
        let gnorm = norm_l2(&g);
        if gnorm < tol {
            return OptResult {
                fun: f(&x),
                x,
                niter: iter,
                converged: true,
            };
        }
        for i in 0..n {
            x[i] -= lr * g[i];
        }
    }

    OptResult {
        fun: f(&x),
        x,
        niter: max_iter,
        converged: false,
    }
}

/// BFGS quasi-Newton method with Armijo line search.
///
/// Maintains an approximate inverse Hessian `H` (initialized to identity) and
/// updates it with the BFGS rank-2 formula at each step.
pub fn minimize_bfgs(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> OptResult {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad(&x);

    // H = inverse Hessian approximation, stored as dense n×n in row-major.
    // Initialized to identity.
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        h[i * n + i] = 1.0;
    }

    for iter in 0..max_iter {
        let gnorm = norm_l2(&g);
        if gnorm < tol {
            return OptResult {
                fun: f(&x),
                x,
                niter: iter,
                converged: true,
            };
        }

        // Search direction: d = -H * g
        let mut d = vec![0.0; n];
        for i in 0..n {
            let row: Vec<f64> = (0..n).map(|j| h[i * n + j] * g[j]).collect();
            d[i] = -binned_sum_f64(&row);
        }

        // Armijo line search.
        let alpha = armijo_line_search(f, &x, &d, &g, 1.0, 1e-4, 0.5);

        // Step: s = alpha * d
        let s: Vec<f64> = d.iter().map(|&di| alpha * di).collect();
        let mut x_new = vec![0.0; n];
        for i in 0..n {
            x_new[i] = x[i] + s[i];
        }

        let g_new = grad(&x_new);

        // y = g_new - g
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(&gn, &go)| gn - go).collect();

        let sy = dot(&s, &y);

        // Skip update if curvature condition is not met (sy too small).
        if sy > f64::EPSILON {
            // BFGS update:
            // H_new = (I - rho*s*y^T) * H * (I - rho*y*s^T) + rho*s*s^T
            // where rho = 1/sy
            let rho = 1.0 / sy;

            // Compute H*y
            let mut hy = vec![0.0; n];
            for i in 0..n {
                let row: Vec<f64> = (0..n).map(|j| h[i * n + j] * y[j]).collect();
                hy[i] = binned_sum_f64(&row);
            }

            let yhy = dot(&y, &hy);

            // Update H in-place using the Sherman-Morrison-Woodbury form:
            // H_new = H - (H*y*s^T + s*y^T*H) * rho + (1 + rho*y^T*H*y) * rho * s*s^T
            let factor = (1.0 + rho * yhy) * rho;
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] = h[i * n + j]
                        - rho * (hy[i] * s[j] + s[i] * hy[j])
                        + factor * s[i] * s[j];
                }
            }
        }

        x = x_new;
        g = g_new;
    }

    OptResult {
        fun: f(&x),
        x,
        niter: max_iter,
        converged: false,
    }
}

/// L-BFGS (limited-memory BFGS) with Armijo line search.
///
/// Uses at most `m` recent (s, y) pairs to approximate the inverse Hessian-vector
/// product via the two-loop recursion. Memory usage is O(m*n) instead of O(n^2).
pub fn minimize_lbfgs(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    m: usize,
    tol: f64,
    max_iter: usize,
) -> OptResult {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad(&x);

    // History ring buffer.
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    for iter in 0..max_iter {
        let gnorm = norm_l2(&g);
        if gnorm < tol {
            return OptResult {
                fun: f(&x),
                x,
                niter: iter,
                converged: true,
            };
        }

        // Two-loop recursion to compute d = -H_k * g.
        let k = s_hist.len();
        let mut q = g.clone();
        let mut alpha_vals = vec![0.0; k];

        // First loop: newest to oldest.
        for i in (0..k).rev() {
            alpha_vals[i] = rho_hist[i] * dot(&s_hist[i], &q);
            for j in 0..n {
                q[j] -= alpha_vals[i] * y_hist[i][j];
            }
        }

        // Initial Hessian scaling: H0 = gamma * I
        // gamma = s^T y / y^T y for the most recent pair.
        let gamma = if k > 0 {
            let sy = dot(&s_hist[k - 1], &y_hist[k - 1]);
            let yy = dot(&y_hist[k - 1], &y_hist[k - 1]);
            if yy > f64::EPSILON { sy / yy } else { 1.0 }
        } else {
            1.0
        };

        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

        // Second loop: oldest to newest.
        for i in 0..k {
            let beta = rho_hist[i] * dot(&y_hist[i], &r);
            for j in 0..n {
                r[j] += s_hist[i][j] * (alpha_vals[i] - beta);
            }
        }

        // d = -r (the search direction)
        let d: Vec<f64> = r.iter().map(|&ri| -ri).collect();

        // Armijo line search.
        let alpha = armijo_line_search(f, &x, &d, &g, 1.0, 1e-4, 0.5);

        let s: Vec<f64> = d.iter().map(|&di| alpha * di).collect();
        let mut x_new = vec![0.0; n];
        for i in 0..n {
            x_new[i] = x[i] + s[i];
        }

        let g_new = grad(&x_new);
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(&gn, &go)| gn - go).collect();
        let sy = dot(&s, &y);

        if sy > f64::EPSILON {
            // Add to history, evicting oldest if at capacity.
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }

        x = x_new;
        g = g_new;
    }

    OptResult {
        fun: f(&x),
        x,
        niter: max_iter,
        converged: false,
    }
}

/// Nelder-Mead simplex method (derivative-free).
///
/// Constructs an initial simplex around `x0` and iteratively transforms it using
/// reflection, expansion, contraction, and shrinkage operations.
pub fn minimize_nelder_mead(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> OptResult {
    let n = x0.len();

    // Standard Nelder-Mead parameters.
    let alpha_reflect = 1.0;
    let gamma_expand = 2.0;
    let rho_contract = 0.5;
    let sigma_shrink = 0.5;

    // Build initial simplex: x0 plus n vertices offset along each axis.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        // Use a perturbation that works for both zero and non-zero components.
        let delta = if v[i].abs() > f64::EPSILON {
            v[i] * 0.05
        } else {
            0.00025
        };
        v[i] += delta;
        simplex.push(v);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for iter in 0..max_iter {
        // Sort simplex by function value (deterministic: uses total_cmp for ties).
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| fvals[a].total_cmp(&fvals[b]));
        simplex = indices.iter().map(|&i| simplex[i].clone()).collect();
        fvals = indices.iter().map(|&i| fvals[i]).collect();

        // Check convergence: spread of function values across simplex.
        let f_best = fvals[0];
        let f_worst = fvals[n];
        if (f_worst - f_best).abs() < tol {
            return OptResult {
                x: simplex[0].clone(),
                fun: f_best,
                niter: iter,
                converged: true,
            };
        }

        // Compute centroid of all vertices except the worst.
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection.
        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha_reflect * (centroid[j] - simplex[n][j]))
            .collect();
        let fr = f(&xr);

        if fr < fvals[0] {
            // Try expansion.
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma_expand * (xr[j] - centroid[j]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            // Accept reflection.
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Contraction.
            let (xc, fc) = if fr < fvals[n] {
                // Outside contraction.
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + rho_contract * (xr[j] - centroid[j]))
                    .collect();
                let fc = f(&xc);
                (xc, fc)
            } else {
                // Inside contraction.
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + rho_contract * (simplex[n][j] - centroid[j]))
                    .collect();
                let fc = f(&xc);
                (xc, fc)
            };

            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                // Shrink: move all vertices towards the best.
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + sigma_shrink * (simplex[i][j] - simplex[0][j]);
                    }
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    // Sort one final time to return the best.
    let mut best_idx = 0;
    for i in 1..=n {
        if fvals[i] < fvals[best_idx] {
            best_idx = i;
        }
    }

    OptResult {
        x: simplex[best_idx].clone(),
        fun: fvals[best_idx],
        niter: max_iter,
        converged: false,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod optimize_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y - x^2)^2
    /// Minimum at (1, 1) with f = 0.
    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        binned_sum_f64(&[a * a, 100.0 * b * b])
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let dx = -2.0 * (1.0 - x[0]) + 200.0 * (x[1] - x[0] * x[0]) * (-2.0 * x[0]);
        let dy = 200.0 * (x[1] - x[0] * x[0]);
        vec![dx, dy]
    }

    /// Simple quadratic: f(x) = sum(x_i^2).  Minimum at origin.
    fn quadratic(x: &[f64]) -> f64 {
        let sq: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        binned_sum_f64(&sq)
    }

    fn quadratic_grad(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| 2.0 * xi).collect()
    }

    // -----------------------------------------------------------------------
    // Scalar Root Finding
    // -----------------------------------------------------------------------

    #[test]
    fn test_bisect_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let root = bisect(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_bisect_invalid_bracket() {
        let f = |x: f64| x * x + 1.0; // always positive
        let result = bisect(&f, 0.0, 2.0, 1e-12, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_bisect_determinism() {
        let f = |x: f64| x * x - 2.0;
        let r1 = bisect(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        let r2 = bisect(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    #[test]
    fn test_brentq_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let root = brentq(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_brentq_invalid_bracket() {
        let f = |x: f64| x * x + 1.0;
        let result = brentq(&f, 0.0, 2.0, 1e-12, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_brentq_determinism() {
        let f = |x: f64| x * x - 2.0;
        let r1 = brentq(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        let r2 = brentq(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    #[test]
    fn test_newton_scalar_sqrt4() {
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;
        let root = newton_scalar(&f, &df, 3.0, 1e-12, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_newton_scalar_negative_root() {
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;
        let root = newton_scalar(&f, &df, -3.0, 1e-12, 100).unwrap();
        assert!((root - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_newton_scalar_determinism() {
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;
        let r1 = newton_scalar(&f, &df, 3.0, 1e-12, 100).unwrap();
        let r2 = newton_scalar(&f, &df, 3.0, 1e-12, 100).unwrap();
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    #[test]
    fn test_secant_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let root = secant(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_secant_determinism() {
        let f = |x: f64| x * x - 2.0;
        let r1 = secant(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        let r2 = secant(&f, 1.0, 2.0, 1e-12, 100).unwrap();
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    // -----------------------------------------------------------------------
    // Unconstrained Optimization
    // -----------------------------------------------------------------------

    #[test]
    fn test_minimize_gd_quadratic() {
        let res = minimize_gd(&quadratic, &quadratic_grad, &[5.0, -3.0, 2.0], 0.1, 1000, 1e-8);
        assert!(res.converged);
        for &xi in &res.x {
            assert!(xi.abs() < 1e-3);
        }
    }

    #[test]
    fn test_minimize_gd_rosenbrock() {
        // GD is slow on Rosenbrock, so we use many iterations and a loose tolerance.
        let res = minimize_gd(&rosenbrock, &rosenbrock_grad, &[-1.0, 1.0], 0.001, 100_000, 1e-6);
        // May or may not converge fully, but should get reasonably close.
        assert!((res.x[0] - 1.0).abs() < 0.5);
        assert!((res.x[1] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_minimize_gd_determinism() {
        let r1 = minimize_gd(&quadratic, &quadratic_grad, &[5.0, -3.0], 0.1, 500, 1e-8);
        let r2 = minimize_gd(&quadratic, &quadratic_grad, &[5.0, -3.0], 0.1, 500, 1e-8);
        assert_eq!(r1.x.len(), r2.x.len());
        for (a, b) in r1.x.iter().zip(r2.x.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(r1.fun.to_bits(), r2.fun.to_bits());
    }

    #[test]
    fn test_minimize_bfgs_rosenbrock() {
        let res = minimize_bfgs(&rosenbrock, &rosenbrock_grad, &[-1.0, 1.0], 1e-8, 500);
        assert!(res.converged, "BFGS did not converge on Rosenbrock");
        assert!(
            (res.x[0] - 1.0).abs() < 1e-4,
            "x[0]={} not near 1.0",
            res.x[0]
        );
        assert!(
            (res.x[1] - 1.0).abs() < 1e-4,
            "x[1]={} not near 1.0",
            res.x[1]
        );
        assert!(res.fun < 1e-8, "f(x)={} not near 0", res.fun);
    }

    #[test]
    fn test_minimize_bfgs_quadratic() {
        let res = minimize_bfgs(&quadratic, &quadratic_grad, &[10.0, -7.0, 3.0], 1e-10, 200);
        assert!(res.converged);
        for &xi in &res.x {
            assert!(xi.abs() < 1e-5);
        }
    }

    #[test]
    fn test_minimize_bfgs_determinism() {
        let r1 = minimize_bfgs(&rosenbrock, &rosenbrock_grad, &[-1.0, 1.0], 1e-8, 500);
        let r2 = minimize_bfgs(&rosenbrock, &rosenbrock_grad, &[-1.0, 1.0], 1e-8, 500);
        for (a, b) in r1.x.iter().zip(r2.x.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(r1.fun.to_bits(), r2.fun.to_bits());
        assert_eq!(r1.niter, r2.niter);
    }

    #[test]
    fn test_minimize_lbfgs_rosenbrock() {
        let res = minimize_lbfgs(&rosenbrock, &rosenbrock_grad, &[-1.0, 1.0], 10, 1e-8, 500);
        assert!(res.converged, "L-BFGS did not converge on Rosenbrock");
        assert!(
            (res.x[0] - 1.0).abs() < 1e-4,
            "x[0]={} not near 1.0",
            res.x[0]
        );
        assert!(
            (res.x[1] - 1.0).abs() < 1e-4,
            "x[1]={} not near 1.0",
            res.x[1]
        );
    }

    #[test]
    fn test_minimize_lbfgs_determinism() {
        let r1 = minimize_lbfgs(&rosenbrock, &rosenbrock_grad, &[-1.0, 1.0], 5, 1e-8, 300);
        let r2 = minimize_lbfgs(&rosenbrock, &rosenbrock_grad, &[-1.0, 1.0], 5, 1e-8, 300);
        for (a, b) in r1.x.iter().zip(r2.x.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(r1.niter, r2.niter);
    }

    #[test]
    fn test_minimize_nelder_mead_quadratic() {
        let res = minimize_nelder_mead(&quadratic, &[5.0, -3.0, 2.0], 1e-10, 5000);
        assert!(res.converged, "Nelder-Mead did not converge on quadratic");
        for &xi in &res.x {
            assert!(
                xi.abs() < 1e-3,
                "x_i={} not near 0",
                xi
            );
        }
        assert!(res.fun < 1e-6, "f(x)={} not near 0", res.fun);
    }

    #[test]
    fn test_minimize_nelder_mead_2d() {
        // f(x,y) = (x-3)^2 + (y+1)^2, minimum at (3, -1)
        let f = |x: &[f64]| {
            let a = x[0] - 3.0;
            let b = x[1] + 1.0;
            binned_sum_f64(&[a * a, b * b])
        };
        let res = minimize_nelder_mead(&f, &[0.0, 0.0], 1e-10, 5000);
        assert!(res.converged);
        assert!((res.x[0] - 3.0).abs() < 1e-4);
        assert!((res.x[1] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_minimize_nelder_mead_determinism() {
        let r1 = minimize_nelder_mead(&quadratic, &[5.0, -3.0], 1e-10, 2000);
        let r2 = minimize_nelder_mead(&quadratic, &[5.0, -3.0], 1e-10, 2000);
        for (a, b) in r1.x.iter().zip(r2.x.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(r1.fun.to_bits(), r2.fun.to_bits());
        assert_eq!(r1.niter, r2.niter);
    }

    // -----------------------------------------------------------------------
    // Armijo line search (indirect testing via BFGS)
    // -----------------------------------------------------------------------

    #[test]
    fn test_armijo_decreases_function() {
        // Verify that Armijo returns a step that decreases the function.
        let x = &[2.0, 3.0];
        let g = quadratic_grad(x);
        let d: Vec<f64> = g.iter().map(|&gi| -gi).collect();
        let f0 = quadratic(x);
        let alpha = armijo_line_search(&quadratic, x, &d, &g, 1.0, 1e-4, 0.5);
        let x_new: Vec<f64> = x.iter().zip(d.iter()).map(|(&xi, &di)| xi + alpha * di).collect();
        let f1 = quadratic(&x_new);
        assert!(f1 < f0, "Armijo did not decrease: f0={}, f1={}", f0, f1);
    }
}
