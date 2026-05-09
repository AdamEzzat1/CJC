//! Per-node Bayesian Linear Regression (BLR) head — Phase 0.3b.
//!
//! Naming note (Phase 0.4 Track C-2.3.7): the BLR head is per-*node*,
//! not per-*leaf*. `set_blr_prior` initializes the root's `BlrState`
//! from the prior; every `add_node` / `force_grow` / `force_split`
//! initializes a fresh `BlrState` on the new node. Pre-0.4 docs called
//! this "per-leaf"; the code has always been per-node.
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

use std::cell::RefCell;

use cjc_repro::{KahanAccumulatorF64, KahanAccumulatorF64x4};
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
    /// `blr_update` was given a non-finite (NaN, +Inf, -Inf) value in
    /// either the features matrix or the y vector. Rejected at the
    /// boundary before any posterior update; preserves Welford-style
    /// reproducibility across re-runs.
    NonFiniteInput { value: f64 },
    /// Phase 0.4 Track C-2.3.5 — `blr_update` was called when the
    /// per-node MLP params hash no longer matches the BLR state's
    /// `feature_version_hash`. The MLP that produced the features the
    /// BLR was trained on has changed; continuing the update would
    /// train the posterior on a feature space inconsistent with its
    /// stored mean/precision. Recovery: call `abng_reset_blr` to
    /// re-prime the posterior on the new feature space.
    FeatureVersionStale {
        /// Hash recorded on the BLR state (last init / reset).
        stored: [u8; 32],
        /// Hash of the current per-node MLP params at the call site.
        current: [u8; 32],
    },
    /// Phase 0.4 Track C-2.3.8 — `blr_predict_with_fallback` walked
    /// the parent chain from the requested node up to the root and
    /// found no ancestor with `n_seen >= 1`. The graph has no
    /// observations on the relevant path; predicting from the prior
    /// alone would be uninformative, so the call errors instead of
    /// silently returning prior moments. `walked` counts ancestors
    /// visited (incl. the requested node itself and the root).
    NoEvidence { walked: u32 },
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
            BlrError::NonFiniteInput { value } => write!(
                f,
                "abng blr: input value {value} must be finite (rejected NaN/+Inf/-Inf)"
            ),
            BlrError::FeatureVersionStale { .. } => write!(
                f,
                "abng blr: MLP params changed since BLR posterior was \
                 initialized — feature space is stale. Call \
                 `abng_reset_blr(node_id)` to re-prime the posterior \
                 on the new MLP, then continue training."
            ),
            BlrError::NoEvidence { walked } => write!(
                f,
                "abng blr: no ancestor in {walked}-node parent chain \
                 has any observations (n_seen == 0 everywhere); \
                 fallback predict has no evidence to fall back to"
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
    /// Phase 0.4 Track C-2.3.5 — SHA-256 of the per-node MLP params at
    /// the moment this BLR state was initialized or reset. The graph
    /// layer's `blr_update` rejects when the current params hash
    /// differs (returns `BlrError::FeatureVersionStale`), preventing
    /// the BLR posterior from training on a feature space that no
    /// longer matches the MLP that produced it.
    ///
    /// Set by the graph layer on install (`set_blr_prior`), per-child
    /// initialization (`add_node`), and explicit reset (`reset_blr`).
    /// `BlrState::from_prior` initializes this to all-zeros (sentinel
    /// "uninitialized features"); callers that construct a `BlrState`
    /// without going through the graph layer must set it manually.
    pub feature_version_hash: [u8; 32],
    /// Phase 0.8 Item D1 — cached lower-triangular Cholesky factor
    /// `L` of `precision`. Populated by [`BlrState::update`] and
    /// [`BlrState::combine`] (which already compute it for their own
    /// solves) and consumed by [`BlrState::predict`], which would
    /// otherwise recompute Cholesky from scratch on every call
    /// (`O(d^3 / 3)` flops).
    ///
    /// **NOT part of `canonical_bytes`.** This field is purely a
    /// performance cache; the on-disk snapshot does not include it,
    /// the SHA-256 chain does not depend on it, and any
    /// (de)serialization round-trip leaves it as `None`. The first
    /// `predict` after replay pays the Cholesky cost once and
    /// repopulates the cache.
    ///
    /// **Determinism.** `update` and `combine` produce bit-identical
    /// `mean` / `precision` / `a` / `b` / `n_seen` /
    /// `feature_version_hash` to the pre-D1 code (the cache is set as
    /// a side effect, not a math change). `predict` returns a tuple
    /// computed from `L` — bit-identical regardless of whether `L`
    /// came from the cache or from a fresh Cholesky on the same
    /// `precision`, because `cholesky(precision)` is a deterministic
    /// function.
    ///
    /// **Concurrency.** `RefCell` provides interior mutability so
    /// `predict(&self, ...)` can populate the cache on miss. This is
    /// safe in single-threaded use (which is the contract of the
    /// per-thread arena from C3) and in the `route_to_leaf_batch_par`
    /// path (which never touches BLR state). `BlrState` was already
    /// `!Sync` (transitively, via `cjc_runtime::Tensor`'s `Rc`); the
    /// `RefCell` does not change that.
    ///
    /// **Invariant.** `cached_l` is valid iff it equals
    /// `cholesky(self.precision)`. Methods that mutate `precision`
    /// (`update`, `combine`) must set `cached_l` to the freshly
    /// computed factor. Methods that don't mutate `precision` must
    /// not touch `cached_l` (other than to populate it on lazy
    /// init).
    pub cached_l: RefCell<Option<Vec<f64>>>,
}

impl BlrState {
    /// Initial state from a prior: `mean = 0`, `precision = λ_0 · I`,
    /// `a = a_0`, `b = b_0`. `feature_version_hash` is initialized to
    /// all-zeros (sentinel "uninitialized features"); the graph layer
    /// sets it to the per-node `params_hash` immediately after this
    /// constructor returns.
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
            feature_version_hash: [0u8; 32],
            // Phase 0.8 D1 — first predict will lazy-populate.
            cached_l: RefCell::new(None),
        }
    }

    /// Apply a Normal-Inverse-Gamma conjugate update from a batch.
    ///
    /// `features` is row-major `[n, d]`; `y` is `[n]`. Both as flat slices
    /// for callability from any dispatch path.
    ///
    /// Returns `Ok(None)` on a normal update; `Ok(Some(b_pre_clamp))`
    /// when the post-update `b` was clamped to `f64::EPSILON` to keep
    /// the InverseGamma posterior well-defined (Phase 0.4 Track
    /// C-2.3.4). The graph layer uses this to emit a deterministic
    /// `BlrNumericalRescue` audit event after the corresponding
    /// `BlrUpdated`. The clamped `self.b` is identical with or without
    /// observability, so determinism is preserved.
    pub fn update(&mut self, features: &[f64], y: &[f64]) -> Result<Option<f64>, BlrError> {
        let d = self.d as usize;
        let n = y.len();
        if features.len() != n * d {
            return Err(BlrError::BatchSizeMismatch {
                features_n: (features.len() / d.max(1)) as u32,
                y_n: n as u32,
            });
        }

        // Reject non-finite inputs before any state mutation. Welford /
        // Kahan accumulators silently propagate NaN forever once admitted,
        // and the audit chain would still hash to a stable but-poisoned
        // value, so silent corruption survives replay. Strict reject here
        // matches the BlrError::FeatureDimMismatch boundary-validation
        // style.
        for &v in features {
            if !v.is_finite() {
                return Err(BlrError::NonFiniteInput { value: v });
            }
        }
        for &v in y {
            if !v.is_finite() {
                return Err(BlrError::NonFiniteInput { value: v });
            }
        }

        // Phase 0.8c v14 Item D2b — SIMD-friendly Kahan accumulation.
        //
        // X^T X (d × d) and X^T y (d) and y^T y all reduce over the
        // row dimension `n`. Replace the per-entry scalar Kahan with
        // 4-lane lane-parallel Kahan (`KahanAccumulatorF64x4` from
        // `cjc-repro`), processing 4 rows at a time via `add_lanes`,
        // with leftover rows (n % 4) folded into lane 0 sequentially.
        //
        // # Determinism
        //
        // For n ∈ {1, 2, 3, 4} the result is bit-identical to the
        // pre-D2b scalar Kahan path: the `add_slice` tail walks
        // lane 0 in scalar Kahan order, and `finalize`'s horizontal
        // reduce processes zero-valued lanes as no-ops. For n ≥ 5
        // the lane distribution produces a different (but equally
        // Kahan-stable, and bit-deterministic on every platform)
        // rounding pattern — bit-equal across runs, bit-different
        // from the pre-D2b path. Workloads that go through
        // `Graph::train_step` (n = 1) are unaffected; batched
        // workloads with n ≥ 5 see new BLR `state_hash`es, which
        // propagate through the chain.
        //
        // # Win
        //
        // ~3× on Cholesky-friendly inner-loop math at d ≥ 8. At d=4
        // (the default for most demos) the body is small enough
        // that the wins are marginal in practice; the value is the
        // structural readiness for d=16+ workloads (PINN training,
        // tabular GP with rich basis expansions).
        let mut xtx: Vec<KahanAccumulatorF64x4> =
            vec![KahanAccumulatorF64x4::new(); d * d];
        let mut xty: Vec<KahanAccumulatorF64x4> =
            vec![KahanAccumulatorF64x4::new(); d];
        let mut yty_acc = KahanAccumulatorF64x4::new();

        // Process 4 rows per chunk: each per-entry accumulator sees
        // 4 lane-distributed contributions per chunk.
        let mut i = 0;
        while i + 4 <= n {
            let row0 = &features[(i) * d..(i + 1) * d];
            let row1 = &features[(i + 1) * d..(i + 2) * d];
            let row2 = &features[(i + 2) * d..(i + 3) * d];
            let row3 = &features[(i + 3) * d..(i + 4) * d];
            let y0 = y[i];
            let y1 = y[i + 1];
            let y2 = y[i + 2];
            let y3 = y[i + 3];

            yty_acc.add_lanes([y0 * y0, y1 * y1, y2 * y2, y3 * y3]);
            for a in 0..d {
                xty[a].add_lanes([
                    row0[a] * y0,
                    row1[a] * y1,
                    row2[a] * y2,
                    row3[a] * y3,
                ]);
                for b in 0..d {
                    xtx[a * d + b].add_lanes([
                        row0[a] * row0[b],
                        row1[a] * row1[b],
                        row2[a] * row2[b],
                        row3[a] * row3[b],
                    ]);
                }
            }
            i += 4;
        }
        // Tail: remaining rows fold into lane 0 via `add_slice` on a
        // one-element slice. Each call advances lane 0's running
        // Kahan state by one term; the rest of the lanes stay at
        // whatever the chunk-of-4 loop left them at.
        while i < n {
            let row = &features[i * d..(i + 1) * d];
            let yi = y[i];
            yty_acc.add_slice(&[yi * yi]);
            for a in 0..d {
                xty[a].add_slice(&[row[a] * yi]);
                for b in 0..d {
                    xtx[a * d + b].add_slice(&[row[a] * row[b]]);
                }
            }
            i += 1;
        }

        // Λ_new = Λ + X^T X (in place)
        let mut lambda_new = self.precision.to_vec();
        for a in 0..d {
            for b in 0..d {
                lambda_new[a * d + b] += xtx[a * d + b].finalize();
            }
        }

        // Phase 0.8c v14 Item D3 — fused matvec kernel for the rhs
        // computation: `rhs = Λ_old · μ_old + xty`. Extracted to the
        // free helper [`matvec_plus_xty_kahan`] so the SIMD path
        // (gated at d >= 8 with d % 4 == 0) and the scalar fallback
        // share a single test surface. At d=4 the helper takes the
        // scalar path — bit-identical to the pre-D3 inline code.
        let lambda_old = self.precision.to_vec();
        let mu_old = self.mean.to_vec();
        let xty_finalized: Vec<f64> = xty.iter().map(|x| x.finalize()).collect();
        let rhs = matvec_plus_xty_kahan(&lambda_old, &mu_old, &xty_finalized, d);

        // Solve Λ_new · m_new = rhs via Cholesky.
        let l = cholesky(&lambda_new, d)?;
        let m_new = cholesky_solve(&l, d, &rhs);

        // y^T y was accumulated above. μ_old^T Λ_old μ_old and
        // m_new^T Λ_new m_new are computed via `quadratic_form`
        // (still scalar Kahan — small reductions, no n-axis
        // benefit).
        let yty = yty_acc.finalize();

        let mu_lmu_old = quadratic_form(&lambda_old, d, &mu_old);
        let m_lm_new = quadratic_form(&lambda_new, d, &m_new);

        let a_new = self.a + (n as f64) * 0.5;
        // b_new = b_old + 0.5 (μ_old^T Λ_old μ_old + y^T y - m_new^T Λ_new m_new)
        let b_pre_clamp = self.b + 0.5 * (mu_lmu_old + yty - m_lm_new);
        // Phase 0.4 Track C-2.3.4 — clamp surfaces as a diagnostic
        // audit event at the graph layer. The clamped value is
        // bit-identical with or without observability, so determinism
        // is preserved across runs whether or not callers inspect the
        // returned `Option<f64>`.
        let (b_new, rescue) = if b_pre_clamp < f64::EPSILON {
            (f64::EPSILON, Some(b_pre_clamp))
        } else {
            (b_pre_clamp, None)
        };

        self.mean = Tensor::from_vec(m_new, &[d]).expect("blr mean update");
        self.precision = Tensor::from_vec(lambda_new, &[d, d]).expect("blr precision update");
        self.a = a_new;
        self.b = b_new;
        self.n_seen = self.n_seen.saturating_add(n as u64);
        // Phase 0.8 D1 — `l` is the Cholesky factor of `lambda_new`,
        // which is now `self.precision`. Cache it so subsequent
        // `predict` calls skip the redundant decomposition.
        *self.cached_l.borrow_mut() = Some(l);
        Ok(rescue)
    }

    /// Predict at a single feature vector `phi` of length `d`. Returns
    /// `(mean, epistemic_leverage, aleatoric_var)`.
    ///
    /// * `mean = μ^T φ` — posterior mean prediction.
    /// * `epistemic_leverage = φ^T Λ^(-1) φ = ‖L^(-1) φ‖²` —
    ///   **dimensionless leverage**, NOT variance in y-units. Decreases
    ///   monotonically with evidence (more data → tighter posterior →
    ///   lower leverage at any fixed φ). Naming corrected in Phase 0.4
    ///   Track C-2.3.1; pre-0.4 docs called this slot `epistemic_var`,
    ///   which was misleading because units of variance would require
    ///   multiplying by `aleatoric_var`.
    /// * `aleatoric_var = b / (a - 1)` if `a > 1`, else `f64::INFINITY` —
    ///   mean of the InverseGamma noise variance, in y² units.
    ///
    /// To recover predictive variance in y-units a caller multiplies:
    /// `total_var = aleatoric_var * (1.0 + epistemic_leverage)`.
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

        // epistemic_leverage = φ^T Λ^(-1) φ. Solve Λ x = φ via Cholesky,
        // then φ^T x. Since Λ = L L^T, φ^T Λ^(-1) φ = φ^T L^(-T) L^(-1) φ
        //                                            = ‖L^(-1) φ‖².
        //
        // Phase 0.8 D1 — try the cached factor first. On miss
        // (lazy init after deserialize, or first call before any
        // update) compute fresh and populate the cache. Bit-identical
        // either way: `cholesky(precision)` is deterministic.
        //
        // The two-step extract-then-match pattern is required because
        // the outer `borrow()` and the inner `borrow_mut()` would
        // otherwise conflict at runtime: `RefCell` panics on nested
        // mutable-while-shared. Cloning the `Option<Vec<f64>>` ends
        // the outer borrow before the match arm runs.
        let cached = self.cached_l.borrow().clone();
        let l = match cached {
            Some(l) => l,
            None => {
                let prec = self.precision.to_vec();
                let fresh = cholesky(&prec, self.d as usize)?;
                *self.cached_l.borrow_mut() = Some(fresh.clone());
                fresh
            }
        };
        let z = forward_subst(&l, self.d as usize, phi);
        let mut zsq = KahanAccumulatorF64::new();
        for &zi in &z {
            zsq.add(zi * zi);
        }
        let epistemic_leverage = zsq.finalize();

        let aleatoric_var = if self.a > 1.0 {
            self.b / (self.a - 1.0)
        } else {
            f64::INFINITY
        };
        Ok((mean, epistemic_leverage, aleatoric_var))
    }

    /// Combine `self` (the "into" posterior) with `other` (the
    /// "absorbed" posterior) given the shared `prior`. Phase 0.4 Track
    /// B-2.2.6 — used by `force_merge` (and the policy-driven Merge
    /// fired from `decide_step`) so the merged node inherits absorbed's
    /// evidence instead of dropping it.
    ///
    /// The combine math (per Phase 0.4 prompt §2.2.6):
    /// - `Λ_into ← Λ_into + Λ_other` (sum precisions)
    /// - `m_into ← Λ_new⁻¹ (Λ_into · m_into + Λ_other · m_other)`
    ///   (precision-weighted mean of the two posterior means)
    /// - `a_into ← a_into + a_other - a_prior` (subtract prior to avoid
    ///   double-counting the InverseGamma prior contribution)
    /// - `b_into ← b_into + b_other - b_prior` (same; clamped to
    ///   `f64::EPSILON` if it would go subnormal — same numerical floor
    ///   as `update`, but combine is internal-only so no audit event
    ///   is emitted for the rescue here)
    /// - `n_seen ← n_seen + other.n_seen`
    /// - `feature_version_hash` is unchanged — the merge produces a
    ///   posterior conditioned on `into`'s feature space (the user is
    ///   responsible for ensuring the two MLPs are compatible; the
    ///   `feature_version_hash` mismatch check in `blr_update` does not
    ///   gate `combine` because the combine itself is the authority on
    ///   what feature space the merged posterior lives in).
    ///
    /// Errors only on Cholesky failure (the combined precision must be
    /// positive-definite — true by construction since both inputs were
    /// positive-definite and we summed them).
    pub fn combine(&mut self, other: &BlrState, prior: &BlrPrior) -> Result<(), BlrError> {
        if self.d != other.d {
            return Err(BlrError::FeatureDimMismatch {
                expected: self.d,
                got: other.d,
            });
        }
        let d = self.d as usize;
        let lambda_into_old = self.precision.to_vec();
        let m_into_old = self.mean.to_vec();
        let lambda_other = other.precision.to_vec();
        let m_other = other.mean.to_vec();

        // Λ_new = Λ_into + Λ_other  (per prompt — over-counts prior
        // by one factor; documented approximation, see §2.2.6).
        let mut lambda_new = vec![0.0; d * d];
        for i in 0..d * d {
            lambda_new[i] = lambda_into_old[i] + lambda_other[i];
        }

        // rhs = Λ_into · m_into + Λ_other · m_other  (Kahan-summed).
        let mut rhs = vec![0.0f64; d];
        for i in 0..d {
            let mut acc = KahanAccumulatorF64::new();
            for j in 0..d {
                acc.add(lambda_into_old[i * d + j] * m_into_old[j]);
                acc.add(lambda_other[i * d + j] * m_other[j]);
            }
            rhs[i] = acc.finalize();
        }

        // Solve Λ_new · m_new = rhs via Cholesky.
        let l = cholesky(&lambda_new, d)?;
        let m_new = cholesky_solve(&l, d, &rhs);

        // a, b combine with prior subtract (avoids double-counting the
        // Inverse-Gamma prior contribution).
        let a_new = self.a + other.a - prior.a;
        let mut b_new = self.b + other.b - prior.b;
        if b_new < f64::EPSILON {
            b_new = f64::EPSILON;
        }

        self.precision =
            Tensor::from_vec(lambda_new, &[d, d]).expect("combined precision tensor");
        self.mean = Tensor::from_vec(m_new, &[d]).expect("combined mean tensor");
        self.a = a_new;
        self.b = b_new;
        self.n_seen = self.n_seen.saturating_add(other.n_seen);
        // feature_version_hash unchanged — into's feature space wins.
        // Phase 0.8 D1 — `l` is the Cholesky factor of `lambda_new`,
        // which is now `self.precision`. Cache it for predict.
        *self.cached_l.borrow_mut() = Some(l);
        Ok(())
    }

    /// KL divergence `KL[N(self.mean, self.precision⁻¹) ‖ N(other.mean,
    /// other.precision⁻¹)]` — the Gaussian part of the NIG posterior
    /// only. Used by Phase 0.4 Track B-2.2.3's KL-merge gate inside
    /// `decide_step`. The IG noise component is intentionally omitted;
    /// Merge gates on weight-distribution similarity, not noise-model
    /// agreement.
    ///
    /// Closed form (per prompt §2.2.3, rewritten in terms of precisions
    /// `Λ_i = Σ_i⁻¹` to match `BlrState`'s storage):
    /// ```text
    ///   KL = ½ ( log|Λ_1| − log|Λ_2| − d
    ///            + tr(Λ_2 · Λ_1⁻¹)
    ///            + (m_2 − m_1)ᵀ Λ_2 (m_2 − m_1) )
    /// ```
    ///
    /// Errors on dim mismatch or Cholesky failure on either precision
    /// matrix (both should always be positive-definite by NIG-update
    /// construction; an error here signals a corrupt state).
    pub fn kl_divergence(&self, other: &BlrState) -> Result<f64, BlrError> {
        if self.d != other.d {
            return Err(BlrError::FeatureDimMismatch {
                expected: self.d,
                got: other.d,
            });
        }
        let d = self.d as usize;
        let prec1 = self.precision.to_vec();
        let prec2 = other.precision.to_vec();
        let l1 = cholesky(&prec1, d)?;
        let l2 = cholesky(&prec2, d)?;

        // log|Λ_1| − log|Λ_2| = 2 ∑ (log L_1_ii − log L_2_ii)
        let mut logdet_acc = KahanAccumulatorF64::new();
        for i in 0..d {
            logdet_acc.add(2.0 * l1[i * d + i].ln());
            logdet_acc.add(-2.0 * l2[i * d + i].ln());
        }
        let logdet = logdet_acc.finalize();

        // tr(Λ_2 · Λ_1⁻¹) — solve Λ_1 X = Λ_2 column by column, sum
        // diagonal of X. Symmetric Λ_2 means columns == rows.
        let mut trace_acc = KahanAccumulatorF64::new();
        for col in 0..d {
            let mut col_data = vec![0.0; d];
            for row in 0..d {
                col_data[row] = prec2[row * d + col];
            }
            let solved = cholesky_solve(&l1, d, &col_data);
            trace_acc.add(solved[col]);
        }
        let trace = trace_acc.finalize();

        // (m_2 − m_1)ᵀ Λ_2 (m_2 − m_1)
        let m1 = self.mean.to_vec();
        let m2 = other.mean.to_vec();
        let dm: Vec<f64> = (0..d).map(|i| m2[i] - m1[i]).collect();
        let quad = quadratic_form(&prec2, d, &dm);

        Ok(0.5 * (logdet - d as f64 + trace + quad))
    }

    /// Canonical bytes for hashing. Layout (Phase 0.4 / snapshot v9):
    /// ```text
    ///   d                       u32 BE              (4)
    ///   mean                    f64 BE × d
    ///   precision               f64 BE × d²
    ///   a                       f64 BE              (8)
    ///   b                       f64 BE              (8)
    ///   n_seen                  u64 BE              (8)
    ///   feature_version_hash    [u8; 32]            (32)   ← v9 add (C-2.3.5)
    /// ```
    /// Pre-v9 layout omitted `feature_version_hash`. The new field is
    /// part of the SHA-256 input, so any pre-0.4 `state_hash` is
    /// distinct from its v9 equivalent — clean break, no shadowing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let d = self.d as usize;
        let mut out = Vec::with_capacity(4 + d * 8 + d * d * 8 + 24 + 32);
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
        out.extend_from_slice(&self.feature_version_hash);
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

/// Phase 0.8c v14 Item D3 — fused matvec-plus-bias kernel for the
/// BLR rhs computation.
///
/// Computes `rhs[a] = Σ_b lambda[a*d + b] · mu[b] + xty[a]` for
/// every output index `a ∈ 0..d`, using Kahan-compensated summation
/// throughout. Two execution paths share the same API:
///
/// * **Scalar path** — strict left-to-right Kahan over `(b=0, 1,
///   ..., d-1, xty[a])`. Fires at `d ≤ 4`, at `d ≥ 5 && d % 4 != 0`,
///   and as the regression-safety fallback for any future d shape
///   that doesn't match the SIMD gate. **Bit-identical to the
///   pre-D3 inline code.**
/// * **F64x4 fused path** — lane-parallel Kahan over chunks of 4
///   b-values via [`KahanAccumulatorF64x4::add_lanes`], then a
///   horizontal reduce, then a single outer-Kahan add of `xty[a]`.
///   Fires at `d ≥ 8 && d % 4 == 0`. Bit-different from scalar
///   Kahan in this regime, but determinism is preserved (the F64x4
///   lane assignment + finalize order is byte-stable cross-platform).
///
/// # Why the d=4 gate is `d ≥ 8` not `d ≥ 4`
///
/// At d=4 the SIMD path would introduce a second Kahan stage:
/// `inner_simd_kahan(p0, p1, p2, p3)` produces a scalar that's then
/// fed through an outer Kahan along with `xty[a]`. The pre-D3
/// inline code Kahan-accumulates all five terms (`p0..p3, xty[a]`)
/// inside a single accumulator, threading compensation across all
/// five adds. The two paths produce mathematically equivalent
/// results but typically different f64 bits — and the d=4 demos
/// (every currently locked canary in the repo) would shift. Gating
/// the SIMD path at `d ≥ 8` keeps d=4 bit-identical while still
/// providing the headroom for PINN-scale workloads (d=16+).
///
/// # Inputs
///
/// * `lambda` — row-major `d×d` matrix.
/// * `mu` — length-`d` vector.
/// * `xty_finalized` — length-`d` already-Kahan-summed `X^T y`.
/// * `d` — common dimension (≥ 0; `d=0` returns an empty vec).
///
/// # Panics
///
/// Indexing panics if `lambda.len() < d*d`, `mu.len() < d`, or
/// `xty_finalized.len() < d`. Callers inside `BlrState::update`
/// always satisfy these by construction.
pub(crate) fn matvec_plus_xty_kahan(
    lambda: &[f64],
    mu: &[f64],
    xty_finalized: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut rhs = vec![0.0f64; d];
    let use_simd = d >= 8 && d % 4 == 0;
    for a in 0..d {
        let mut acc = KahanAccumulatorF64::new();
        if use_simd {
            // Lane-parallel Kahan over b ∈ 0..d in chunks of 4.
            // Each chunk distributes 4 products across the F64x4's
            // lanes; `finalize` horizontally reduces into a single
            // scalar.
            let mut acc_x4 = KahanAccumulatorF64x4::new();
            let mut b = 0;
            while b + 4 <= d {
                acc_x4.add_lanes([
                    lambda[a * d + b] * mu[b],
                    lambda[a * d + b + 1] * mu[b + 1],
                    lambda[a * d + b + 2] * mu[b + 2],
                    lambda[a * d + b + 3] * mu[b + 3],
                ]);
                b += 4;
            }
            // No tail: the `use_simd` gate guarantees `d % 4 == 0`.
            acc.add(acc_x4.finalize());
        } else {
            // Strict left-to-right Kahan, matching the pre-D3
            // inline code byte-for-byte at d ≤ 4 and at any d that
            // doesn't qualify for the SIMD path.
            for b in 0..d {
                acc.add(lambda[a * d + b] * mu[b]);
            }
        }
        acc.add(xty_finalized[a]);
        rhs[a] = acc.finalize();
    }
    rhs
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
    fn epistemic_leverage_decreases_with_evidence() {
        let p = BlrPrior::new(0.01, 1.0, 1.0).unwrap();
        let mut s = BlrState::from_prior(&p, 2);
        let phi = [1.0, 0.5];
        let (_m0, lev0, _) = s.predict(&phi).unwrap();
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
        let (_m1, lev1, _) = s.predict(&phi).unwrap();
        assert!(
            lev1 < lev0,
            "epistemic leverage didn't decrease: {lev0} → {lev1}"
        );
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
        // v9 layout (Phase 0.4 Track C-2.3.5):
        // 4 + 4*8 + 16*8 + 8 + 8 + 8 + 32 = 220
        assert_eq!(s.canonical_bytes().len(), 220);
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

    // ── Phase 0.4 Track C-2.3.4: numerical-rescue audit observability ──

    #[test]
    fn update_returns_none_on_normal_path() {
        // Healthy prior + non-degenerate update → no clamp, no rescue.
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut s = BlrState::from_prior(&p, 2);
        let rescue = s.update(&[1.0, 2.0, 0.5, 1.5], &[1.0, 1.5]).unwrap();
        assert!(rescue.is_none(), "normal update should not trigger rescue");
        assert!(s.b >= f64::EPSILON);
    }

    #[test]
    fn update_returns_some_when_b_clamps_to_epsilon() {
        // Degenerate prior: `b = ε / 2` already below threshold. Any
        // update with SSR == 0 keeps b_pre_clamp == b_old < ε so the
        // clamp fires deterministically.
        //   y = [0], X = [[1]] with prior μ = 0 ⇒ m_new = 0,
        //   yty = 0, mu_lmu_old = 0, m_lm_new = 0 ⇒ SSR = 0.
        let half_eps = f64::EPSILON / 2.0;
        let p = BlrPrior::new(1.0, 1.0, half_eps).unwrap();
        let mut s = BlrState::from_prior(&p, 1);
        let rescue = s.update(&[1.0], &[0.0]).unwrap();
        let pre = rescue.expect("clamp should fire on degenerate prior");
        assert_eq!(pre.to_bits(), half_eps.to_bits());
        assert_eq!(s.b, f64::EPSILON, "post-clamp b is ε");
    }

    #[test]
    fn rescue_is_deterministic_across_runs() {
        // Two independent runs with identical inputs hit the clamp with
        // bit-identical pre-clamp values — the rescue path must
        // preserve the determinism guarantee.
        let half_eps = f64::EPSILON / 2.0;
        let p = BlrPrior::new(1.0, 1.0, half_eps).unwrap();
        let mut a = BlrState::from_prior(&p, 1);
        let mut b = BlrState::from_prior(&p, 1);
        let r_a = a.update(&[1.0], &[0.0]).unwrap();
        let r_b = b.update(&[1.0], &[0.0]).unwrap();
        assert_eq!(r_a, r_b);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn audit_kind_blr_numerical_rescue_tag_is_0x18() {
        let kind = crate::audit::AuditKind::BlrNumericalRescue {
            reason: crate::audit::BLR_RESCUE_B_BELOW_EPSILON,
            b_pre_clamp_bits: 0u64,
        };
        assert_eq!(kind.tag(), 0x18);
    }

    // ── Phase 0.4 Track B-2.2.6: NIG-aware merge math ─────────────────

    #[test]
    fn combine_dim_mismatch_errs() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut a = BlrState::from_prior(&p, 2);
        let b = BlrState::from_prior(&p, 3);
        let err = a.combine(&b, &p).unwrap_err();
        assert!(matches!(err, BlrError::FeatureDimMismatch { .. }));
    }

    #[test]
    fn combine_two_priors_yields_uniform_posterior() {
        // Two unupdated priors combined: precision sums (2 × λ I),
        // mean stays 0, a = 2a-a = a, b = 2b-b = b.
        let p = BlrPrior::new(1.0, 2.0, 3.0).unwrap();
        let mut a = BlrState::from_prior(&p, 2);
        let b = BlrState::from_prior(&p, 2);
        a.combine(&b, &p).unwrap();
        // Precision diagonal is 2*1.0 = 2.0; off-diagonal 0.
        let prec = a.precision.to_vec();
        assert_eq!(prec[0], 2.0);
        assert_eq!(prec[1], 0.0);
        assert_eq!(prec[2], 0.0);
        assert_eq!(prec[3], 2.0);
        // Mean stays at 0.
        for &x in a.mean.to_vec().iter() {
            assert_eq!(x, 0.0);
        }
        // a, b restored to prior after the +other-prior subtract.
        assert_eq!(a.a, 2.0);
        assert_eq!(a.b, 3.0);
        assert_eq!(a.n_seen, 0);
    }

    #[test]
    fn combine_increases_n_seen() {
        // Combining two posteriors that have each been updated once
        // (n_seen=1 each) gives n_seen=2.
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut left = BlrState::from_prior(&p, 1);
        let mut right = BlrState::from_prior(&p, 1);
        left.update(&[1.0], &[2.0]).unwrap();
        right.update(&[2.0], &[1.0]).unwrap();
        assert_eq!(left.n_seen, 1);
        assert_eq!(right.n_seen, 1);
        left.combine(&right, &p).unwrap();
        assert_eq!(left.n_seen, 2);
    }

    #[test]
    fn combine_is_deterministic() {
        // Two independent runs of the same combine produce
        // bit-identical posterior bytes.
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut a1 = BlrState::from_prior(&p, 2);
        let mut b1 = BlrState::from_prior(&p, 2);
        a1.update(&[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        b1.update(&[2.0, 1.0], &[2.0]).unwrap();

        let mut a2 = BlrState::from_prior(&p, 2);
        let mut b2 = BlrState::from_prior(&p, 2);
        a2.update(&[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        b2.update(&[2.0, 1.0], &[2.0]).unwrap();

        a1.combine(&b1, &p).unwrap();
        a2.combine(&b2, &p).unwrap();
        assert_eq!(a1.canonical_bytes(), a2.canonical_bytes());
    }

    // ── Phase 0.4 Track B-2.2.3: KL divergence ────────────────────────

    #[test]
    fn kl_divergence_self_is_zero() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let s = BlrState::from_prior(&p, 3);
        let kl = s.kl_divergence(&s).unwrap();
        assert!(kl.abs() < 1e-10, "KL[X || X] should be 0, got {kl}");
    }

    #[test]
    fn kl_divergence_two_priors_equal_is_zero() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let a = BlrState::from_prior(&p, 2);
        let b = BlrState::from_prior(&p, 2);
        let kl = a.kl_divergence(&b).unwrap();
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn kl_divergence_grows_with_mean_separation() {
        // Two priors that have been updated differently — their
        // posteriors now have different means and the KL should be
        // non-zero and grow as the means diverge.
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut a = BlrState::from_prior(&p, 2);
        let mut b1 = BlrState::from_prior(&p, 2);
        let mut b2 = BlrState::from_prior(&p, 2);
        a.update(&[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        b1.update(&[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        b2.update(&[1.0, 0.5, 0.5, 1.0], &[100.0, 100.0]).unwrap();
        let kl_close = a.kl_divergence(&b1).unwrap();
        let kl_far = a.kl_divergence(&b2).unwrap();
        assert!(kl_close.abs() < 1e-10);
        assert!(kl_far > 0.0);
        assert!(
            kl_far > kl_close,
            "KL should grow with separation: close={kl_close}, far={kl_far}"
        );
    }

    #[test]
    fn kl_divergence_dim_mismatch_errs() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let a = BlrState::from_prior(&p, 2);
        let b = BlrState::from_prior(&p, 3);
        let err = a.kl_divergence(&b).unwrap_err();
        assert!(matches!(err, BlrError::FeatureDimMismatch { .. }));
    }

    #[test]
    fn kl_divergence_is_deterministic() {
        let p = BlrPrior::new(1.0, 1.0, 1.0).unwrap();
        let mut a1 = BlrState::from_prior(&p, 2);
        let mut b1 = BlrState::from_prior(&p, 2);
        a1.update(&[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        b1.update(&[2.0, 1.0], &[2.0]).unwrap();
        let kl1 = a1.kl_divergence(&b1).unwrap();

        let mut a2 = BlrState::from_prior(&p, 2);
        let mut b2 = BlrState::from_prior(&p, 2);
        a2.update(&[1.0, 0.5, 0.5, 1.0], &[1.0, 1.0]).unwrap();
        b2.update(&[2.0, 1.0], &[2.0]).unwrap();
        let kl2 = a2.kl_divergence(&b2).unwrap();
        assert_eq!(kl1.to_bits(), kl2.to_bits());
    }

    #[test]
    fn combine_clamps_b_when_negative() {
        // Construct a degenerate scenario: prior b is large, posteriors
        // somehow have small b. Subtracting prior gives negative; clamp
        // fires. Use direct field manipulation since the conjugate
        // update math itself shouldn't drive b below prior.
        let p = BlrPrior::new(1.0, 1.0, 100.0).unwrap();
        let mut a = BlrState::from_prior(&p, 1);
        let mut b = BlrState::from_prior(&p, 1);
        a.b = 1.0;
        b.b = 1.0;
        a.combine(&b, &p).unwrap();
        // a.b + b.b - p.b = 1 + 1 - 100 = -98; clamped to ε.
        assert_eq!(a.b, f64::EPSILON);
    }

    // ── Phase 0.8c v14 Item D3: matvec_plus_xty_kahan ────────────────

    /// Scalar reference implementation: strict left-to-right Kahan,
    /// identical to the pre-D3 inline body. Used as the bit-identity
    /// oracle in the tests below.
    fn matvec_plus_xty_kahan_scalar_reference(
        lambda: &[f64],
        mu: &[f64],
        xty_finalized: &[f64],
        d: usize,
    ) -> Vec<f64> {
        let mut rhs = vec![0.0f64; d];
        for a in 0..d {
            let mut acc = KahanAccumulatorF64::new();
            for b in 0..d {
                acc.add(lambda[a * d + b] * mu[b]);
            }
            acc.add(xty_finalized[a]);
            rhs[a] = acc.finalize();
        }
        rhs
    }

    #[test]
    fn matvec_plus_xty_d4_bit_identical_to_scalar() {
        // The canary-preserving guarantee at d=4: helper output ==
        // pre-D3 inline output, bit-for-bit, for any input.
        let lambda = vec![
            2.0, -1.0, 0.5, 0.25,
            -1.0, 3.0, 0.1, -0.2,
            0.5, 0.1, 2.5, 0.3,
            0.25, -0.2, 0.3, 1.8,
        ];
        let mu = vec![0.7, -0.4, 1.1, 0.05];
        let xty = vec![0.1, -0.2, 0.3, -0.05];
        let got = matvec_plus_xty_kahan(&lambda, &mu, &xty, 4);
        let want = matvec_plus_xty_kahan_scalar_reference(&lambda, &mu, &xty, 4);
        for i in 0..4 {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "D3 helper at d=4 must be bit-identical to scalar at index {i}"
            );
        }
    }

    #[test]
    fn matvec_plus_xty_d1_d2_d3_bit_identical() {
        // The full scalar regime: d ∈ {1, 2, 3} also stays
        // bit-identical to the pre-D3 inline body.
        for d in 1..=3usize {
            let lambda: Vec<f64> = (0..d * d).map(|i| ((i + 1) as f64) * 0.5).collect();
            let mu: Vec<f64> = (0..d).map(|i| ((i + 1) as f64) * 0.7).collect();
            let xty: Vec<f64> = (0..d).map(|i| ((i + 1) as f64) * 0.05).collect();
            let got = matvec_plus_xty_kahan(&lambda, &mu, &xty, d);
            let want = matvec_plus_xty_kahan_scalar_reference(&lambda, &mu, &xty, d);
            for i in 0..d {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "d={d}, i={i}");
            }
        }
    }

    #[test]
    fn matvec_plus_xty_d8_simd_path_within_kahan_tolerance() {
        // At d=8 the SIMD path activates. Result must still match
        // the scalar reference within Kahan-stability tolerance
        // (the two paths thread compensation differently but should
        // agree on the final value to many ULPs).
        let lambda: Vec<f64> = (0..64).map(|i| ((i as f64) * 0.13).sin()).collect();
        let mu: Vec<f64> = (0..8).map(|i| ((i as f64) * 0.31).cos()).collect();
        let xty: Vec<f64> = (0..8).map(|i| ((i as f64) * 0.07).sin()).collect();
        let got = matvec_plus_xty_kahan(&lambda, &mu, &xty, 8);
        let want = matvec_plus_xty_kahan_scalar_reference(&lambda, &mu, &xty, 8);
        for i in 0..8 {
            let abs_err = (got[i] - want[i]).abs();
            assert!(
                abs_err < 1e-14,
                "d=8 SIMD path diverged from scalar at i={i}: got={}, want={}, |diff|={abs_err:.3e}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn matvec_plus_xty_d5_d6_d7_take_scalar_fallback() {
        // The SIMD gate is `d >= 8 && d % 4 == 0`. d ∈ {5, 6, 7}
        // fall through to scalar, which must be bit-identical to
        // the reference.
        for d in 5..=7usize {
            let lambda: Vec<f64> = (0..d * d).map(|i| ((i as f64) * 0.11).sin()).collect();
            let mu: Vec<f64> = (0..d).map(|i| ((i as f64) * 0.23).cos()).collect();
            let xty: Vec<f64> = (0..d).map(|i| ((i as f64) * 0.17).sin()).collect();
            let got = matvec_plus_xty_kahan(&lambda, &mu, &xty, d);
            let want = matvec_plus_xty_kahan_scalar_reference(&lambda, &mu, &xty, d);
            for i in 0..d {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "d={d}, i={i}: scalar fallback should be bit-identical"
                );
            }
        }
    }

    #[test]
    fn matvec_plus_xty_d10_d14_take_scalar_fallback() {
        // d = 10 and 14: d ≥ 8 but d % 4 ≠ 0, so scalar fallback
        // fires. Bit-identical to reference.
        for &d in &[10usize, 14] {
            let lambda: Vec<f64> = (0..d * d).map(|i| ((i as f64) * 0.13).cos()).collect();
            let mu: Vec<f64> = (0..d).map(|i| ((i as f64) * 0.29).sin()).collect();
            let xty: Vec<f64> = (0..d).map(|i| ((i as f64) * 0.11).cos()).collect();
            let got = matvec_plus_xty_kahan(&lambda, &mu, &xty, d);
            let want = matvec_plus_xty_kahan_scalar_reference(&lambda, &mu, &xty, d);
            for i in 0..d {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "d={d}, i={i}: d%4≠0 must take scalar fallback"
                );
            }
        }
    }

    #[test]
    fn matvec_plus_xty_d0_returns_empty() {
        // Degenerate but well-defined: d=0 → empty vec.
        let lambda: Vec<f64> = vec![];
        let mu: Vec<f64> = vec![];
        let xty: Vec<f64> = vec![];
        let got = matvec_plus_xty_kahan(&lambda, &mu, &xty, 0);
        assert!(got.is_empty());
    }

    #[test]
    fn matvec_plus_xty_d16_simd_deterministic_across_runs() {
        // Same input twice → identical bytes. Pins the
        // SIMD-path-is-deterministic contract at the largest d
        // most likely to be exercised by PINN workloads.
        let lambda: Vec<f64> = (0..256).map(|i| ((i as f64) * 0.019).sin()).collect();
        let mu: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.041).cos()).collect();
        let xty: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.013).sin()).collect();
        let a = matvec_plus_xty_kahan(&lambda, &mu, &xty, 16);
        let b = matvec_plus_xty_kahan(&lambda, &mu, &xty, 16);
        for i in 0..16 {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "i={i}");
        }
    }
}
