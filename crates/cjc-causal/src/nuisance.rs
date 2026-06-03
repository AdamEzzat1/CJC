//! Linear nuisance estimators for double machine learning.
//!
//! v0.1 ships **linear nuisances via OLS** rather than neural-network
//! nuisances. The DML headline is the orthogonal-moment + cross-fitting
//! discipline, not the flexibility of the nuisance. Linear DML is the
//! canonical textbook implementation and the byte-identity reproducibility
//! contract is much tighter for OLS than for MLP training (where init
//! scheme, epoch count, lr, and cross-platform RNG all need to be fixed).
//! MLP nuisances via `cjc_ad::GradGraph` are a v0.2 extension.
//!
//! Pipeline per fold: fit `lm()` on the training subset → manually compute
//! predictions on the test subset using `α + β'·x`. Both steps are
//! Kahan-summed.

use crate::error::CausalError;
use cjc_repro::KahanAccumulatorF64;

/// Fit OLS on `(x_train, y_train)` and predict on `x_test`.
///
/// Both `x_train` and `x_test` are row-major n×p matrices (without the
/// intercept column — `cjc_runtime::hypothesis::lm` adds it internally).
///
/// Returns predictions for each row of `x_test`. Returns `CausalError::Numerical`
/// if the OLS solver fails (e.g. rank-deficient training design matrix).
pub fn fit_linear_predict(
    x_train: &[f64],
    y_train: &[f64],
    x_test: &[f64],
    p: usize,
) -> Result<Vec<f64>, CausalError> {
    let n_train = y_train.len();
    let n_test = x_test.len() / p;
    if x_train.len() != n_train * p {
        return Err(CausalError::Numerical {
            detail: format!(
                "nuisance fit: x_train shape ({}) doesn't match n_train * p ({}*{})",
                x_train.len(),
                n_train,
                p
            ),
        });
    }
    if x_test.len() != n_test * p {
        return Err(CausalError::Numerical {
            detail: format!(
                "nuisance predict: x_test shape ({}) doesn't match n_test * p ({}*{})",
                x_test.len(),
                n_test,
                p
            ),
        });
    }

    let fit = cjc_runtime::hypothesis::lm(x_train, y_train, n_train, p)
        .map_err(|e| CausalError::Numerical { detail: format!("nuisance OLS failed: {}", e) })?;

    // Predict: ŷ_i = β_0 + Σ β_{j+1}·x_test[i, j]
    // fit.coefficients = [β_0 (intercept), β_1, ..., β_p]
    let mut preds = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let mut acc = KahanAccumulatorF64::new();
        acc.add(fit.coefficients[0]);
        for j in 0..p {
            acc.add(fit.coefficients[j + 1] * x_test[i * p + j]);
        }
        preds.push(acc.finalize());
    }
    Ok(preds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_predict_recovers_perfect_linear_relationship() {
        // y = 1 + 2*x_1 + 3*x_2 (exactly), with x_1 and x_2 LINEARLY
        // INDEPENDENT (otherwise the (β_1, β_2) split isn't identified and
        // out-of-sample predictions diverge from the true model). We use
        // x_1 = i, x_2 = i^2 — not collinear.
        let n_train = 20;
        let n_test = 5;
        let p = 2;
        let mut x_train = Vec::with_capacity(n_train * p);
        let mut y_train = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let x1 = i as f64;
            let x2 = (i * i) as f64;
            x_train.push(x1);
            x_train.push(x2);
            y_train.push(1.0 + 2.0 * x1 + 3.0 * x2);
        }
        let mut x_test = Vec::with_capacity(n_test * p);
        for i in 0..n_test {
            // Use x_1 and x_2 values that also respect the non-collinear
            // shape (close to training distribution).
            x_test.push(5.0 + i as f64);
            x_test.push((5.0 + i as f64).powi(2));
        }
        let preds = fit_linear_predict(&x_train, &y_train, &x_test, p).unwrap();
        for i in 0..n_test {
            let x1 = 5.0 + i as f64;
            let x2 = x1 * x1;
            let expected = 1.0 + 2.0 * x1 + 3.0 * x2;
            assert!(
                (preds[i] - expected).abs() < 1e-6,
                "row {}: expected {}, got {}",
                i,
                expected,
                preds[i]
            );
        }
    }

    #[test]
    fn fit_predict_shape_mismatch_returns_error() {
        let x_train = vec![1.0, 2.0, 3.0]; // 3 elements, n_train=2, p=2 needs 4
        let y_train = vec![1.0, 2.0];
        let x_test = vec![0.0, 0.0];
        assert!(fit_linear_predict(&x_train, &y_train, &x_test, 2).is_err());
    }

    #[test]
    fn fit_predict_on_constant_target_returns_constant_predictions() {
        let n_train = 10;
        let n_test = 3;
        let p = 1;
        let x_train: Vec<f64> = (0..n_train).map(|i| i as f64).collect();
        let y_train: Vec<f64> = vec![5.0; n_train];
        let x_test: Vec<f64> = vec![100.0, 200.0, 300.0];
        let preds = fit_linear_predict(&x_train, &y_train, &x_test, p).unwrap();
        for p_i in &preds {
            assert!((p_i - 5.0).abs() < 1e-9, "expected 5.0, got {}", p_i);
        }
    }
}
