//! Per-node density tracker — Phase 0.3c.
//!
//! Maintains a Welford running mean + diagonal `M2` over the BLR
//! feature space (dimension `d` = penultimate-feature dim). Used by
//! the composite OOD score's `density_score` signal.
//!
//! # Why diagonal, not full covariance
//!
//! Storage `O(d)` instead of `O(d²)`. Mahalanobis becomes a single
//! pass over `d` floats — no Cholesky inversion at score time. For
//! the chess-RL value-head application (post-tanh penultimate features)
//! the covariance is approximately diagonal anyway. Full-covariance
//! tracker is a Phase 0.3d optimisation if specific failure modes
//! demand it.
//!
//! # Determinism
//!
//! All sums use `KahanAccumulatorF64`; canonical bytes via
//! `f64::to_bits().to_be_bytes()`. Bit-deterministic for a fixed
//! observation order.

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::tensor::Tensor;

const VARIANCE_FLOOR: f64 = 1e-12;

/// Errors specific to the density-tracker subsystem.
#[derive(Debug, PartialEq)]
pub enum DensityError {
    /// `set_density_tracker` was called twice on the same graph.
    AlreadyFrozen,
    /// `set_density_tracker` was called before `set_leaf_head` — `d` is unknown.
    NoLeafHead,
    /// `set_density_tracker` was called on a graph that already has child nodes.
    NotEmptyGraph { n_nodes: u32 },
    /// A density op was called on a node without an installed tracker.
    NoDensityTracker,
    /// `density_observe` got features whose second axis didn't match `d`.
    FeatureDimMismatch { expected: u32, got: u32 },
}

impl std::fmt::Display for DensityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DensityError::AlreadyFrozen => write!(f, "abng density: already frozen"),
            DensityError::NoLeafHead => {
                write!(f, "abng density: must be installed *after* the leaf head")
            }
            DensityError::NotEmptyGraph { n_nodes } => write!(
                f,
                "abng density: must be installed before any add_node \
                 (graph already has {n_nodes} nodes)"
            ),
            DensityError::NoDensityTracker => write!(f, "abng density: no tracker installed"),
            DensityError::FeatureDimMismatch { expected, got } => write!(
                f,
                "abng density: features dim {got} doesn't match d={expected}"
            ),
        }
    }
}

/// Per-node feature-density tracker.
#[derive(Debug, Clone)]
pub struct DensityTracker {
    pub d: u32,
    pub n: u64,
    pub mean: Tensor, // [d]
    pub m2: Tensor,   // [d] — diagonal Welford M2
}

impl DensityTracker {
    /// Construct a fresh tracker at dimension `d` with zero observations.
    pub fn new(d: u32) -> Self {
        let dz = d as usize;
        let mean = Tensor::from_vec(vec![0.0; dz], &[dz]).expect("density mean tensor");
        let m2 = Tensor::from_vec(vec![0.0; dz], &[dz]).expect("density m2 tensor");
        Self { d, n: 0, mean, m2 }
    }

    /// Apply a batch of observations. `features` is row-major `[n_obs, d]`.
    pub fn observe_batch(&mut self, features: &[f64]) -> Result<(), DensityError> {
        let dz = self.d as usize;
        if features.len() % dz != 0 {
            return Err(DensityError::FeatureDimMismatch {
                expected: self.d,
                got: features.len() as u32,
            });
        }
        let n_obs = features.len() / dz;
        let mut mean = self.mean.to_vec();
        let mut m2 = self.m2.to_vec();
        let mut n = self.n;
        for row_i in 0..n_obs {
            let row = &features[row_i * dz..(row_i + 1) * dz];
            n += 1;
            let n_f = n as f64;
            for k in 0..dz {
                let delta = row[k] - mean[k];
                mean[k] += delta / n_f;
                let delta2 = row[k] - mean[k];
                m2[k] += delta * delta2;
            }
        }
        self.n = n;
        self.mean = Tensor::from_vec(mean, &[dz]).expect("density mean update");
        self.m2 = Tensor::from_vec(m2, &[dz]).expect("density m2 update");
        Ok(())
    }

    /// Per-dim sample variance, with `0.0` for `n < 2`.
    pub fn variance(&self) -> Vec<f64> {
        let dz = self.d as usize;
        let m2 = self.m2.to_vec();
        if self.n < 2 {
            return vec![0.0; dz];
        }
        let denom = (self.n - 1) as f64;
        m2.iter().map(|&x| x / denom).collect()
    }

    /// Diagonal Mahalanobis squared distance from `phi` to the running
    /// mean. Returns `0.0` when `n < 2` (no signal yet).
    pub fn mahalanobis_squared(&self, phi: &[f64]) -> Result<f64, DensityError> {
        if phi.len() != self.d as usize {
            return Err(DensityError::FeatureDimMismatch {
                expected: self.d,
                got: phi.len() as u32,
            });
        }
        if self.n < 2 {
            return Ok(0.0);
        }
        let mean = self.mean.to_vec();
        let var = self.variance();
        let mut acc = KahanAccumulatorF64::new();
        for i in 0..(self.d as usize) {
            let diff = phi[i] - mean[i];
            let v = var[i].max(VARIANCE_FLOOR);
            acc.add(diff * diff / v);
        }
        Ok(acc.finalize())
    }

    /// Density score `1 − exp(−mahal²)`, in `[0, 1)`. Higher = more OOD.
    pub fn density_score(&self, phi: &[f64]) -> Result<f64, DensityError> {
        let m2 = self.mahalanobis_squared(phi)?;
        Ok(1.0 - (-m2).exp())
    }

    /// Canonical big-endian byte encoding for hashing.
    /// Layout: `d u32 + n u64 + mean f64×d + m2 f64×d`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let dz = self.d as usize;
        let mut out = Vec::with_capacity(4 + 8 + dz * 16);
        out.extend_from_slice(&self.d.to_be_bytes());
        out.extend_from_slice(&self.n.to_be_bytes());
        for x in self.mean.to_vec() {
            out.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        for x in self.m2.to_vec() {
            out.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        out
    }

    /// SHA-256 of canonical bytes.
    pub fn state_hash(&self) -> [u8; 32] {
        cjc_snap::hash::sha256(&self.canonical_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_tracker_zero_n() {
        let t = DensityTracker::new(3);
        assert_eq!(t.n, 0);
        assert_eq!(t.variance(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn welford_recovers_mean() {
        let mut t = DensityTracker::new(2);
        let xs = vec![
            1.0, 2.0, // sample 0
            3.0, 4.0, // sample 1
            5.0, 6.0, // sample 2
        ];
        t.observe_batch(&xs).unwrap();
        assert_eq!(t.n, 3);
        let mean = t.mean.to_vec();
        assert!((mean[0] - 3.0).abs() < 1e-12);
        assert!((mean[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn welford_recovers_variance() {
        let mut t = DensityTracker::new(1);
        // Sequence with known sample variance: [1, 2, 3] → var = 1.0
        t.observe_batch(&[1.0, 2.0, 3.0]).unwrap();
        let var = t.variance();
        assert!((var[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mahalanobis_zero_at_mean() {
        let mut t = DensityTracker::new(2);
        t.observe_batch(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        // mean = (3, 4)
        let mahal = t.mahalanobis_squared(&[3.0, 4.0]).unwrap();
        assert!(mahal.abs() < 1e-9);
    }

    #[test]
    fn mahalanobis_increases_with_distance() {
        let mut t = DensityTracker::new(1);
        t.observe_batch(&[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        // mean = 2, var = 2.5
        let near = t.mahalanobis_squared(&[2.0]).unwrap();
        let mid = t.mahalanobis_squared(&[5.0]).unwrap();
        let far = t.mahalanobis_squared(&[10.0]).unwrap();
        assert!(far > mid);
        assert!(mid > near);
    }

    #[test]
    fn density_score_returns_zero_when_n_lt_two() {
        let t = DensityTracker::new(2);
        let s = t.density_score(&[1.0, 1.0]).unwrap();
        assert_eq!(s, 0.0);
    }

    #[test]
    fn density_score_bounded_in_range() {
        let mut t = DensityTracker::new(1);
        t.observe_batch(&[1.0, 2.0, 3.0]).unwrap();
        // Far-away point → score saturates near 1 (may reach exactly 1.0
        // when exp(-very_large) underflows to 0).
        let s = t.density_score(&[1000.0]).unwrap();
        assert!(s > 0.99 && s <= 1.0);
        // At mean → score = 0
        let s = t.density_score(&[2.0]).unwrap();
        assert!(s.abs() < 1e-9);
        // Moderate distance: score in (0, 1).
        let s = t.density_score(&[3.5]).unwrap();
        assert!(s > 0.0 && s < 1.0);
    }

    #[test]
    fn observe_dim_mismatch_errs() {
        let mut t = DensityTracker::new(2);
        let err = t.observe_batch(&[1.0]).unwrap_err();
        assert!(matches!(err, DensityError::FeatureDimMismatch { expected: 2, .. }));
    }

    #[test]
    fn determinism_double_run() {
        let mk = || {
            let mut t = DensityTracker::new(3);
            for k in 0..50 {
                let phi = [k as f64 * 0.1, (k as f64 * 0.05).sin(), (k as f64).cos()];
                t.observe_batch(&phi).unwrap();
            }
            t
        };
        let a = mk();
        let b = mk();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn canonical_bytes_size() {
        let t = DensityTracker::new(4);
        // 4 (d) + 8 (n) + 4*8 (mean) + 4*8 (m2) = 76
        assert_eq!(t.canonical_bytes().len(), 76);
    }

    #[test]
    fn state_hash_changes_after_observe() {
        let mut t = DensityTracker::new(2);
        let h0 = t.state_hash();
        t.observe_batch(&[1.0, 2.0]).unwrap();
        let h1 = t.state_hash();
        assert_ne!(h0, h1);
    }
}
