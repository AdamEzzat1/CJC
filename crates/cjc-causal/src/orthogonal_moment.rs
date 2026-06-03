//! Chernozhukov 2018 orthogonal moment for the partially linear model.
//!
//! Model: `y = β·T + g(X) + ε`, `T = m(X) + V`, with `E[V·X] = 0` and
//! `E[ε·(T, X)] = 0`. The orthogonal (Neyman) score is constructed so that
//! errors in the nuisance estimators `ĝ` and `m̂` don't blow up the rate
//! for `β̂` faster than `n^{-1/4}` per nuisance.
//!
//! Given cross-fitted predictions `ŷ = ĝ(X)` and `t̂ = m̂(X)` (each row
//! predicted from the fold that did NOT include it during training):
//!
//! ```text
//! e_Y(i) = y(i) - ŷ(i)
//! e_T(i) = T(i) - t̂(i)
//! β̂ = Σ e_T(i)·e_Y(i) / Σ e_T(i)²
//! ε̂(i) = e_Y(i) - β̂·e_T(i)
//! σ̂² = Σ (e_T(i)·ε̂(i))² / n
//! SE(β̂) = sqrt(σ̂² / n) / mean(e_T²)
//! ```
//!
//! All sums above are Kahan-compensated. The output `(β̂, SE)` is therefore
//! a deterministic function of the input vectors plus the cross-fit splits.

use crate::error::CausalError;
use cjc_repro::KahanAccumulatorF64;

/// Result of the orthogonal-moment scoring step.
#[derive(Clone, Debug, PartialEq)]
pub struct PartialLinearScore {
    /// Treatment-effect estimate `β̂`.
    pub beta: f64,
    /// Asymptotic standard error of `β̂` via the plug-in variance estimator.
    pub std_error: f64,
    /// Residual-product mean `(1/n) Σ e_T·e_Y` — surfaced for diagnostics
    /// (it's the numerator of `β̂` divided by `n`).
    pub residual_product_mean: f64,
    /// Mean of squared T-residuals `(1/n) Σ e_T²` — the denominator of `β̂`.
    /// Surfaced for diagnostics; near-zero values indicate that the
    /// treatment is well-predicted by the covariates (low signal for IV-free
    /// causal identification — see ADR-0043 §determinism rule note).
    pub e_t_sq_mean: f64,
}

/// Compute the orthogonal-moment β̂ and standard error for the partially
/// linear DML model.
///
/// # Arguments
///
/// - `y` — observed outcomes (length n)
/// - `t` — observed treatments (length n; continuous is fine)
/// - `y_hat` — cross-fitted predictions of `E[Y|X]` (length n)
/// - `t_hat` — cross-fitted predictions of `E[T|X]` (length n)
///
/// All four slices must have the same length.
///
/// # Errors
///
/// - `CausalError::Numerical` if `Σ e_T² < f64::EPSILON` (the treatment is
///   essentially fully predicted by the covariates — the partial-linear
///   identification fails because there's no residual T-variation to
///   regress Y-residuals against).
pub fn partial_linear_score(
    y: &[f64],
    t: &[f64],
    y_hat: &[f64],
    t_hat: &[f64],
) -> Result<PartialLinearScore, CausalError> {
    let n = y.len();
    if t.len() != n || y_hat.len() != n || t_hat.len() != n {
        return Err(CausalError::Numerical {
            detail: format!(
                "partial_linear_score: length mismatch y={}, t={}, y_hat={}, t_hat={}",
                y.len(), t.len(), y_hat.len(), t_hat.len()
            ),
        });
    }
    if n < 2 {
        return Err(CausalError::Numerical {
            detail: format!("partial_linear_score: need n >= 2, got {}", n),
        });
    }

    // Compute residuals + accumulate the numerator and denominator of β̂.
    let mut e_t = Vec::with_capacity(n);
    let mut e_y = Vec::with_capacity(n);
    let mut num_acc = KahanAccumulatorF64::new();
    let mut den_acc = KahanAccumulatorF64::new();
    for i in 0..n {
        let et = t[i] - t_hat[i];
        let ey = y[i] - y_hat[i];
        num_acc.add(et * ey);
        den_acc.add(et * et);
        e_t.push(et);
        e_y.push(ey);
    }
    let num = num_acc.finalize();
    let den = den_acc.finalize();

    if !(den > f64::EPSILON) {
        return Err(CausalError::Numerical {
            detail: format!(
                "partial_linear_score: Σ (T - t̂)² = {} is too small — \
                 treatment is fully explained by covariates (no residual variation)",
                den
            ),
        });
    }

    let beta = num / den;

    // Plug-in variance: σ̂² = (1/n) Σ (e_T·ε̂)², ε̂ = e_Y - β̂·e_T.
    // SE(β̂) = sqrt(σ̂²/n) / mean(e_T²)
    let mut var_acc = KahanAccumulatorF64::new();
    for i in 0..n {
        let epsilon_hat = e_y[i] - beta * e_t[i];
        let term = e_t[i] * epsilon_hat;
        var_acc.add(term * term);
    }
    let sigma_sq = var_acc.finalize() / n as f64;
    let n_f = n as f64;
    let e_t_sq_mean = den / n_f;
    // The standard DML variance estimator: V(β̂) = σ̂² / (n · (mean(e_T²))²)
    let var_beta = sigma_sq / (n_f * e_t_sq_mean * e_t_sq_mean);
    if !(var_beta >= 0.0) {
        return Err(CausalError::Numerical {
            detail: format!("partial_linear_score: V(β̂) = {} is non-positive", var_beta),
        });
    }
    let std_error = var_beta.sqrt();

    Ok(PartialLinearScore {
        beta,
        std_error,
        residual_product_mean: num / n_f,
        e_t_sq_mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_linear_relationship_recovers_beta() {
        // Construct data where Y = 2*T (no covariate effect, no noise).
        // y_hat and t_hat both perfect (mean-zero residuals from the X-only model).
        // With y_hat = t_hat = 0 (assume covariates explain nothing), e_T = T, e_Y = Y = 2*T.
        // β̂ = Σ T·(2T) / Σ T² = 2.
        let t: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = t.iter().map(|x| 2.0 * x).collect();
        let y_hat: Vec<f64> = vec![0.0; 5];
        let t_hat: Vec<f64> = vec![0.0; 5];
        let score = partial_linear_score(&y, &t, &y_hat, &t_hat).unwrap();
        assert!((score.beta - 2.0).abs() < 1e-12, "got β = {}", score.beta);
    }

    #[test]
    fn zero_residual_variation_returns_error() {
        // y_hat = y exactly, t_hat = t exactly → all residuals are zero.
        let t: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
        let result = partial_linear_score(&y, &t, &y, &t);
        assert!(result.is_err());
    }

    #[test]
    fn length_mismatch_returns_error() {
        let t = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0]; // wrong length
        let y_hat = vec![0.0; 2];
        let t_hat = vec![0.0; 2];
        assert!(partial_linear_score(&y, &t, &y_hat, &t_hat).is_err());
    }

    #[test]
    fn n_too_small_returns_error() {
        let t = vec![1.0];
        let y = vec![2.0];
        let y_hat = vec![0.0];
        let t_hat = vec![0.0];
        assert!(partial_linear_score(&y, &t, &y_hat, &t_hat).is_err());
    }

    #[test]
    fn standard_error_is_non_negative() {
        let t: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<f64> = vec![2.1, 3.9, 6.1, 7.9, 10.1, 11.9]; // ~2x + noise
        let y_hat: Vec<f64> = vec![0.0; 6];
        let t_hat: Vec<f64> = vec![0.0; 6];
        let score = partial_linear_score(&y, &t, &y_hat, &t_hat).unwrap();
        assert!(score.std_error >= 0.0);
    }

    #[test]
    fn shifting_outcome_does_not_change_beta() {
        // Shifting y by a constant should not change β̂ in the orthogonal-
        // moment formula AS LONG AS y_hat is also shifted by the same
        // constant (which is what cross-fitting with a constant feature
        // achieves). Here we shift both — the orthogonal moment is
        // invariant.
        let t: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = vec![1.5, 3.5, 5.5, 7.5, 9.5];
        let y_hat: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let t_hat: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let s1 = partial_linear_score(&y, &t, &y_hat, &t_hat).unwrap();

        let c = 10.0;
        let y_shifted: Vec<f64> = y.iter().map(|x| x + c).collect();
        let y_hat_shifted: Vec<f64> = y_hat.iter().map(|x| x + c).collect();
        let s2 = partial_linear_score(&y_shifted, &t, &y_hat_shifted, &t_hat).unwrap();

        assert!((s1.beta - s2.beta).abs() < 1e-9);
    }

    #[test]
    fn diagnostics_fields_are_populated() {
        let t: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
        let y_hat: Vec<f64> = vec![0.0; 4];
        let t_hat: Vec<f64> = vec![0.0; 4];
        let score = partial_linear_score(&y, &t, &y_hat, &t_hat).unwrap();
        // e_T² mean: (1+4+9+16)/4 = 7.5
        assert!((score.e_t_sq_mean - 7.5).abs() < 1e-12);
        // residual product mean: (2+8+18+32)/4 = 15.0
        assert!((score.residual_product_mean - 15.0).abs() < 1e-12);
        // β = 15.0 / 7.5 = 2.0
        assert!((score.beta - 2.0).abs() < 1e-12);
    }
}
