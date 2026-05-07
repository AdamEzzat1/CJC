//! Per-leaf Bayesian Linear Regression (BLR) head — Phase 0.3b.
//!
//! The "Bayesian last layer" architecture: the MLP from Phase 0.3a serves
//! as a feature extractor; this module attaches a closed-form Bayesian
//! linear regression to the *penultimate* features. The BLR head provides
//! posterior mean + variance for the regression output, separating
//! epistemic uncertainty (from finite training data) from aleatoric
//! uncertainty (irreducible noise).
//!
//! Single-output regression only (Phase 0.3b scope). The output dim is
//! always 1; the input is `d`-dimensional, where `d` is the MLP's
//! penultimate-feature dimension.
//!
//! # Conjugate model
//!
//! ```text
//!   prior:     w | σ²   ~  N(0, σ² · Λ_0^(-1))           with Λ_0 = λ_0 · I
//!              σ²       ~  InvGamma(a_0, b_0)
//!   likelihood: y_i | x_i, w, σ²  ~  N(w^T x_i, σ²)
//!   posterior: w | σ², D ~ N(μ, σ² · Λ^(-1))             [Normal-IG conjugate]
//!              σ² | D    ~ InvGamma(a, b)
//! ```
//!
//! # Determinism
//!
//! All sums use `KahanAccumulatorF64` from `cjc-repro`; Cholesky uses
//! plain `+`/`-`/`*`/`/`/`sqrt` on `f64` with no FMA. Bit-deterministic
//! for fixed input order across runs and platforms.

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::tensor::Tensor;

/// Errors specific to the BLR subsystem.
#[derive(Debug, PartialEq)]
pub enum BlrError {
    /// `set_blr_prior` was called twice on the same graph.
    AlreadyFrozen,
    /// `set_blr_prior` was called before `set_leaf_head` — `d` is unknown.
    NoLeafHead,
    /// `set_blr_prior` was called on a graph that already has child nodes.
    NotEmptyGraph { n_nodes: u32 },
    /// Prior parameters violated `precision > 0`, `a > 0`, or `b > 0`.
    InvalidPrior,
    /// A BLR op was called on a graph without an installed prior.
    NoBlrPrior,
    /// `blr_update` got a features tensor whose second axis didn't match `d`.
    FeatureDimMismatch { expected: u32, got: u32 },
    /// `blr_update` features and y batch sizes didn't match.
    BatchSizeMismatch { features_n: u32, y_n: u32 },
    /// Cholesky decomposition encountered a non-positive pivot — the
    /// matrix is not positive definite. Should not happen with the NIG
    /// math; signals a corrupt state.
    NonPositiveDefinite,
}

impl std::fmt::Display for BlrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlrError::AlreadyFrozen => write!(f, "abng blr: prior already frozen"),
            BlrError::NoLeafHead => write!(
                f,
                "abng blr: prior must be installed *after* the leaf head (which determines d)"
            ),
            BlrError::NotEmptyGraph { n_nodes } => write!(
                f,
                "abng blr: prior must be installed before any add_node \
                 (graph already has {n_nodes} nodes)"
            ),
            BlrError::InvalidPrior => {
                write!(f, "abng blr: prior parameters must satisfy precision > 0, a > 0, b > 0")
            }
            BlrError::NoBlrPrior => write!(f, "abng blr: no prior installed"),
            BlrError::FeatureDimMismatch { expected, got } => write!(
                f,
                "abng blr: features dim {got} doesn't match prior d={expected}"
            ),
            BlrError::BatchSizeMismatch { features_n, y_n } => write!(
                f,
                "abng blr: features batch size {features_n} doesn't match y size {y_n}"
            ),
            BlrError::NonPositiveDefinite => write!(
                f,
                "abng blr: precision matrix is not positive-definite (corrupt state)"
            ),
        }
    }
}

/// Frozen graph-wide BLR prior.
#[derive(Debug, Clone)]
pub struct BlrPrior {
    /// Isotropic precision λ_0 (so `Λ_0 = λ_0 · I`).
    pub precision: f64,
    /// InverseGamma shape `a_0 > 0`.
    pub a: f64,
    /// InverseGamma scale `b_0 > 0`.
    pub b: f64,
    /// SHA-256 of canonical bytes — embedded in the audit witness.
    pub config_hash: [u8; 32],
}

impl BlrPrior {
    /// Construct a fresh `BlrPrior`. Computes `config_hash` deterministically.
    /// Returns `Err(BlrError::InvalidPrior)` if any parameter is non-positive.
    pub fn new(precision: f64, a: f64, b: f64) -> Result<Self, BlrError> {
        if !(precision > 0.0) || !(a > 0.0) || !(b > 0.0) {
            return Err(BlrError::InvalidPrior);
        }
        let mut p = Self {
            precision,
            a,
            b,
            config_hash: [0u8; 32],
        };
        p.config_hash = cjc_snap::hash::sha256(&p.canonical_bytes());
        Ok(p)
    }

    /// Canonical 24-byte big-endian encoding for hashing.
    pub fn canonical_bytes(&self) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[0..8].copy_from_slice(&self.precision.to_bits().to_be_bytes());
        out[8..16].copy_from_slice(&self.a.to_bits().to_be_bytes());
        out[16..24].copy_from_slice(&self.b.to_bits().to_be_bytes());
        out
    }
}

/// Posterior state for a per-node BLR head.
#[derive(Debug, Clone)]
pub struct BlrState {
    /// Penultimate-feature dimension.
    pub d: u32,
    /// Posterior mean of weights, shape `[d]`.
    pub mean: Tensor,
    /// Posterior precision matrix `Λ`, shape `[d, d]` (positive-definite).
    pub precision: Tensor,
    /// Posterior InverseGamma shape `a`.
    pub a: f64,
    /// Posterior InverseGamma scale `b`.
    pub b: f64,
    /// Total observations applied.
    pub n_seen: u64,
}

impl BlrState {
    /// Initial state from a prior: `mean = 0`, `precision = λ_0 · I`,
    /// `a = a_0`, `b = b_0`.
    pub fn from_prior(prior: &BlrPrior, d: u32) -> Self {
        let dz = d as usize;
        let mean = Tensor::from_vec(vec![0.0; dz], &[dz]).expect("blr mean tensor");
        let mut prec_data = vec![0.0; dz * dz];
        for i in 0..dz {
            prec_data[i * dz + i] = prior.precision;
        }
        let precision = Tensor::from_vec(prec_data, &[dz, dz]).expect("blr precision tensor");
        Self {
            d,
            mean,
            precision,
            a: prior.a,
            b: prior.b,
            n_seen: 0,
        }
    }

    /// Apply a Normal-Inverse-Gamma conjugate update from a batch.
    ///
    /// `features` is row-major `[n, d]`; `y` is `[n]`. Both as flat slices
    /// for callability from any dispatch path.
    pub fn update(&mut self, features: &[f64], y: &[f64]) -> Result<(), BlrError> {
        let d = self.d as usize;
        let n = y.len();
        if features.len() != n * d {
            return Err(BlrError::BatchSizeMismatch {
                features_n: (features.len() / d.max(1)) as u32,
                y_n: n as u32,
            });
        }

        // X^T X (d × d) and X^T y (d): both Kahan-accumulated.
        let mut xtx = vec![KahanAccumulatorF64::new(); d * d];
        let mut xty = vec![KahanAccumulatorF64::new(); d];
        for i in 0..n {
            let row = &features[i * d..(i + 1) * d];
            let yi = y[i];
            for a in 0..d {
                xty[a].add(row[a] * yi);
                for b in 0..d {
                    xtx[a * d + b].add(row[a] * row[b]);
                }
            }
        }

        // Λ_new = Λ + X^T X (in place)
        let mut lambda_new = self.precision.to_vec();
        for a in 0..d {
            for b in 0..d {
                lambda_new[a * d + b] += xtx[a * d + b].finalize();
            }
        }

        // rhs = Λ_old · μ_old + X^T y     (length d)
        let lambda_old = self.precision.to_vec();
        let mu_old = self.mean.to_vec();
        let mut rhs = vec![0.0f64; d];
        for a in 0..d {
            let mut acc = KahanAccumulatorF64::new();
            for b in 0..d {
                acc.add(lambda_old[a * d + b] * mu_old[b]);
            }
            acc.add(xty[a].finalize());
            rhs[a] = acc.finalize();
        }

        // Solve Λ_new · m_new = rhs via Cholesky.
        let l = cholesky(&lambda_new, d)?;
        let m_new = cholesky_solve(&l, d, &rhs);

        // y^T y, μ_old^T Λ_old μ_old, m_new^T Λ_new m_new — all Kahan.
        let mut yty_acc = KahanAccumulatorF64::new();
        for &yi in y {
            yty_acc.add(yi * yi);
        }
        let yty = yty_acc.finalize();

        let mu_lmu_old = quadratic_form(&lambda_old, d, &mu_old);
        let m_lm_new = quadratic_form(&lambda_new, d, &m_new);

        let a_new = self.a + (n as f64) * 0.5;
        // b_new = b_old + 0.5 (μ_old^T Λ_old μ_old + y^T y - m_new^T Λ_new m_new)
        let mut b_new = self.b + 0.5 * (mu_lmu_old + yty - m_lm_new);
        if b_new < f64::EPSILON {
            // Numerical floor — keep IG well-defined.
            b_new = f64::EPSILON;
        }

        self.mean = Tensor::from_vec(m_new, &[d]).expect("blr mean update");
        self.precision = Tensor::from_vec(lambda_new, &[d, d]).expect("blr precision update");
        self.a = a_new;
        self.b = b_new;
        self.n_seen = self.n_seen.saturating_add(n as u64);
        Ok(())
    }

    /// Predict at a single feature vector `phi` of length `d`. Returns
    /// `(mean, epistemic_var, aleatoric_var)`.
    ///
    /// * `mean = μ^T φ`
    /// * `epistemic_var = φ^T Λ^(-1) φ` (variance of the posterior mean)
    /// * `aleatoric_var = b / (a - 1)` if `a > 1`, else `f64::INFINITY`
    ///   (mean of the InverseGamma noise variance).
    pub fn predict(&self, phi: &[f64]) -> Result<(f64, f64, f64), BlrError> {
        if phi.len() != self.d as usize {
            return Err(BlrError::FeatureDimMismatch {
                expected: self.d,
                got: phi.len() as u32,
            });
        }
        let mu = self.mean.to_vec();
        let mut mean_acc = KahanAccumulatorF64::new();
        for i in 0..(self.d as usize) {
            mean_acc.add(mu[i] * phi[i]);
        }
        let mean = mean_acc.finalize();

        // epistemic_var = φ^T Λ^(-1) φ. Solve Λ x = φ via Cholesky, then
        // φ^T x. Since Λ = L L^T, φ^T Λ^(-1) φ = φ^T L^(-T) L^(-1) φ
        //                                       = ‖L^(-1) φ‖².
        let prec = self.precision.to_vec();
        let l = cholesky(&prec, self.d as usize)?;
        let z = forward_subst(&l, self.d as usize, phi);
        let mut zsq = KahanAccumulatorF64::new();
        for &zi in &z {
            zsq.add(zi * zi);
        }
        let epistemic_var = zsq.finalize();

        let aleatoric_var = if self.a > 1.0 {
            self.b / (self.a - 1.0)
        } else {
            f64::INFINITY
        };
        Ok((mean, epistemic_var, aleatoric_var))
    }

    /// Canonical bytes for hashing. Layout:
    /// ```text
    ///   d           u32 BE                 (4)
    ///   mean        f64 BE × d
    ///   precision   f64 BE × d²
    ///   a           f64 BE                 (8)
    ///   b           f64 BE                 (8)
    ///   n_seen      u64 BE                 (8)
    /// ```
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let d = self.d as usize;
        let mut out = Vec::with_capacity(4 + d * 8 + d * d * 8 + 24);
        out.extend_from_slice(&self.d.to_be_bytes());
        for x in self.mean.to_vec() {
            out.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        for x in self.precision.to_vec() {
            out.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        out.extend_from_slice(&self.a.to_bits().to_be_bytes());
        out.extend_from_slice(&self.b.to_bits().to_be_bytes());
        out.extend_from_slice(&self.n_seen.to_be_bytes());
        out
    }

    /// SHA-256 of canonical bytes.
    pub fn state_hash(&self) -> [u8; 32] {
        cjc_snap::hash::sha256(&self.canonical_bytes())
    }
}

// ─── Cholesky + triangular solves ─────────────────────────────────────────

/// Compute the lower-triangular Cholesky factor `L` such that
/// `A = L L^T`, given a positive-definite symmetric `A` of dimension `d`.
/// `A` is row-major `d × d`. Returns `L` row-major `d × d` (upper triangle
/// zeroed). Errors if a non-positive pivot is encountered.
pub fn cholesky(a: &[f64], d: usize) -> Result<Vec<f64>, BlrError> {
    debug_assert_eq!(a.len(), d * d);
    let mut l = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut acc = KahanAccumulatorF64::new();
            for k in 0..j {
                acc.add(l[i * d + k] * l[j * d + k]);
            }
            let s = a[i * d + j] - acc.finalize();
            if i == j {
                if s <= 0.0 {
                    return Err(BlrError::NonPositiveDefinite);
                }
                l[i * d + j] = s.sqrt();
            } else {
                let pivot = l[j * d + j];
                if pivot == 0.0 {
                    return Err(BlrError::NonPositiveDefinite);
                }
                l[i * d + j] = s / pivot;
            }
        }
    }
    Ok(l)
}

/// Forward substitution: solve `L y = b` for `y`, given lower-triangular `L`.
fn forward_subst(l: &[f64], d: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; d];
    for i in 0..d {
        let mut acc = KahanAccumulatorF64::new();
        for k in 0..i {
            acc.add(l[i * d + k] * y[k]);
        }
        y[i] = (b[i] - acc.finalize()) / l[i * d + i];
    }
    y
}

/// Back substitution: solve `L^T x = y` for `x`, given lower-triangular `L`.
fn back_subst(l: &[f64], d: usize, y: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0f64; d];
    for ii in (0..d).rev() {
        let mut acc = KahanAccumulatorF64::new();
        for k in (ii + 1)..d {
            acc.add(l[k * d + ii] * x[k]);
        }
        x[ii] = (y[ii] - acc.finalize()) / l[ii * d + ii];
    }
    x
}

/// Solve `A x = b` given the Cholesky factor `L` of `A`. `L` is
/// lower-triangular, row-major `d × d`.
pub fn cholesky_solve(l: &[f64], d: usize, b: &[f64]) -> Vec<f64> {
    let y = forward_subst(l, d, b);
    back_subst(l, d, &y)
}

/// `x^T A x`, with `x: [d]`, `A: [d, d]` row-major. Kahan-summed.
fn quadratic_form(a: &[f64], d: usize, x: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..d {
        for j in 0..d {
            acc.add(x[i] * a[i * d + j] * x[j]);
        }
    }
    acc.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn cholesky_recovers_identity() {
        // A = I → L = I.
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let l = cholesky(&a, 2).unwrap();
        assert_eq!(l, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn cholesky_reconstructs_simple_pd_matrix() {
        // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]]
        // Check L L^T == A
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        // L L^T
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0;
                for k in 0..2 {
                    s += l[i * 2 + k] * l[j * 2 + k];
                }
                assert!(approx_eq(s, a[i * 2 + j], 1e-12));
            }
        }
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        // Indefinite matrix.
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert_eq!(cholesky(&a, 2).unwrap_err(), BlrError::NonPositiveDefinite);
    }

    #[test]
    fn cholesky_solve_recovers_known_solution() {
        // A x = b with A = [[2, 0], [0, 3]], b = [4, 6] → x = [2, 2].
        let a = vec![2.0, 0.0, 0.0, 3.0];
        let l = cholesky(&a, 2).unwrap();
        let x = cholesky_solve(&l, 2, &[4.0, 6.0]);
        assert!(approx_eq(x[0], 2.0, 1e-12));
        assert!(approx_eq(x[1], 2.0, 1e-12));
    }

    #[test]
    fn from_prior_sets_isotropic_precision() {
        let p = BlrPrior::new(2.5, 1.0, 1.0).unwrap();
        let s = BlrState::from_prior(&p, 3);
        let mean = s.mean.to_vec();
        assert_eq!(mean, vec![0.0; 3]);
        let prec = s.precision.to_vec();
        // Diagonal = 2.5, off-diagonal = 0.
        for i in 0..3 {
            for j in 0..3 {
                let exp = if i == j { 2.5 } else { 0.0 };
                assert_eq!(prec[i * 3 + j], exp);
            }
        }
    }

    #[test]
    fn invalid_prior_rejected() {
        assert_eq!(BlrPrior::new(0.0, 1.0, 1.0).unwrap_err(), BlrError::InvalidPrior);
        assert_eq!(BlrPrior::new(1.0, 0.0, 1.0).unwrap_err(), BlrError::InvalidPrior);
        assert_eq!(BlrPrior::new(1.0, 1.0, 0.0).unwrap_err(), BlrError::InvalidPrior);
        assert_eq!(BlrPrior::new(-1.0, 1.0, 1.0).unwrap_err(), BlrError::InvalidPrior);
    }

    #[test]
    fn nig_update_recovers_y_when_data_is_consistent() {
        // d=1: regress y on a single feature x. Use x = [1, 1, 1, 1, 1] (constant)
        // and y = [3, 3, 3, 3, 3]. Posterior mean should approach 3 and
        // residual variance should approach 0.
        let p = BlrPrior::new(0.001, 1.0, 1.0).unwrap();
        let mut s = BlrState::from_prior(&p, 1);
        let xs = vec![1.0; 100];
        let ys = vec![3.0; 100];
        s.update(&xs, &ys).unwrap();
        let mu = s.mean.to_vec()[0];
        assert!((mu - 3.0).abs() < 0.01, "mean drifted: got {mu}");
        // Predict at x=1
        let (m, _epi, _ale) = s.predict(&[1.0]).unwrap();
        assert!((m - 3.0).abs() < 0.01);
    }

    #[test]
    fn nig_update_d2_recovers_known_weights() {
        // Two features with unique linear combination: y = 2 x1 + 3 x2.
        // Generate 200 deterministic points and fit.
        let p = BlrPrior::new(0.001, 1.0, 1.0).unwrap();
        let mut s = BlrState::from_prior(&p, 2);
        let mut xs = Vec::with_capacity(400);
        let mut ys = Vec::with_capacity(200);
        for i in 0..200 {
            let x1 = (i as f64) * 0.01;
            let x2 = ((i + 7) as f64) * 0.013;
            xs.push(x1);
            xs.push(x2);
            ys.push(2.0 * x1 + 3.0 * x2);
        }
        s.update(&xs, &ys).unwrap();
        let mu = s.mean.to_vec();
        assert!((mu[0] - 2.0).abs() < 0.01, "w0 drifted: got {}", mu[0]);
        assert!((mu[1] - 3.0).abs() < 0.01, "w1 drifted: got {}", mu[1]);
    }

    #[test]
    fn epistemic_variance_decreases_with_evidence() {
        let p = BlrPrior::new(0.01, 1.0, 1.0).unwrap();
        let mut s = BlrState::from_prior(&p, 2);
        let phi = [1.0, 0.5];
        let (_m0, epi0, _) = s.predict(&phi).unwrap();
        // Update with 50 informative samples.
        let xs = (0..50)
            .flat_map(|i| {
                let x1 = (i as f64).sin();
                let x2 = (i as f64).cos();
                [x1, x2]
            })
            .collect::<Vec<f64>>();
        let ys = (0..50).map(|i| (i as f64).sin() * 2.0).collect::<Vec<f64>>();
        s.update(&xs, &ys).unwrap();
        let (_m1, epi1, _) = s.predict(&phi).unwrap();
        assert!(epi1 < epi0, "epistemic var didn't decrease: {epi0} → {epi1}");
    }

    #[test]
    fn aleatoric_variance_inf_when_a_le_one() {
        // a_0 = 0.5 < 1 → aleatoric should be +inf before any update.
        let p = BlrPrior::new(1.0, 0.5, 1.0).unwrap();
        let s = BlrState::from_prior(&p, 1);
        let (_m, _e, a) = s.predict(&[1.0]).unwrap();
        assert!(a.is_infinite());
    }

    #[test]
    fn dim_mismatch_in_predict_errs() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let s = BlrState::from_prior(&p, 3);
        let err = s.predict(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, BlrError::FeatureDimMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn batch_size_mismatch_in_update_errs() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut s = BlrState::from_prior(&p, 2);
        // 3 features but only 1 y → mismatch (3 features per row × 1 = 3 ≠ 2 × 1 expected)
        let err = s.update(&[1.0, 2.0, 3.0], &[1.0]).unwrap_err();
        assert!(matches!(err, BlrError::BatchSizeMismatch { .. }));
    }

    #[test]
    fn canonical_bytes_size() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let s = BlrState::from_prior(&p, 4);
        // 4 + 4*8 + 16*8 + 8 + 8 + 8 = 188
        assert_eq!(s.canonical_bytes().len(), 188);
    }

    #[test]
    fn state_hash_changes_after_update() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut s = BlrState::from_prior(&p, 1);
        let h0 = s.state_hash();
        s.update(&[1.0], &[2.0]).unwrap();
        let h1 = s.state_hash();
        assert_ne!(h0, h1);
    }

    #[test]
    fn update_is_deterministic() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut a = BlrState::from_prior(&p, 2);
        let mut b = BlrState::from_prior(&p, 2);
        let xs = vec![1.0, 2.0, 0.5, 1.5, 2.0, 0.3];
        let ys = vec![1.0, 1.5, 2.5];
        a.update(&xs, &ys).unwrap();
        b.update(&xs, &ys).unwrap();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn prior_canonical_bytes_size_24() {
        let p = BlrPrior::new(1.0, 2.0, 3.0).unwrap();
        assert_eq!(p.canonical_bytes().len(), 24);
    }

    #[test]
    fn prior_config_hash_changes_with_params() {
        let a = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let b = BlrPrior::new(1.5, 1.0, 1.0).unwrap();
        assert_ne!(a.config_hash, b.config_hash);
    }
}
