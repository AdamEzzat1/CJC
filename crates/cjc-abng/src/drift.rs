//! Per-node drift baseline + score — Phase 0.3c.
//!
//! A `DriftBaseline` is a frozen snapshot of a node's
//! [`DensityTracker`](crate::density::DensityTracker) at a moment of
//! the user's choosing. Once frozen, the score
//!
//! ```text
//!   shift = (1/d) · Σᵢ ( (μ_current[i] − μ_baseline[i]) / max(σ_baseline[i], ε) )²
//!   drift_score = √shift
//! ```
//!
//! is the per-dim z-shift L2-normalised. Larger means more drift. No
//! threshold is baked in — Phase 0.3d's structural-decision engine
//! sets the cutoff.
//!
//! # Why not the more popular Wasserstein-1 / KL?
//!
//! For diagonal-Welford trackers, exact W1/KL between two Gaussians
//! reduces to a per-dim mean+variance comparison anyway. The L2
//! z-shift is the cheaper, equally informative signal that doesn't
//! require knowing both running variances. Phase 0.3d can refine if
//! a specific failure mode demands it.
//!
//! # Determinism
//!
//! All sums Kahan-compensated. Canonical bytes via
//! `f64::to_bits().to_be_bytes()`. Bit-deterministic.

use cjc_repro::KahanAccumulatorF64;
use cjc_runtime::tensor::Tensor;

use crate::density::{DensityError, DensityTracker};

const STD_FLOOR: f64 = 1e-12;

/// Errors specific to the drift-detector subsystem.
#[derive(Debug, PartialEq)]
pub enum DriftError {
    /// `freeze_drift_baseline` was called on a node without a density tracker.
    NoDensityTracker,
    /// `freeze_drift_baseline` was called when the density tracker had `n < 2`
    /// observations — variance isn't defined.
    InsufficientEvidence { n: u64 },
    /// `drift_score` was called on a node without a frozen baseline.
    NoBaseline,
    /// Density and baseline dimensions disagree (corrupt state).
    DimMismatch { density_d: u32, baseline_d: u32 },
}

impl std::fmt::Display for DriftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriftError::NoDensityTracker => write!(
                f,
                "abng drift: density tracker must be installed before freezing baseline"
            ),
            DriftError::InsufficientEvidence { n } => write!(
                f,
                "abng drift: density tracker needs n ≥ 2 observations to freeze baseline (got n={n})"
            ),
            DriftError::NoBaseline => write!(f, "abng drift: no baseline frozen"),
            DriftError::DimMismatch {
                density_d,
                baseline_d,
            } => write!(
                f,
                "abng drift: density d={density_d} doesn't match baseline d={baseline_d}"
            ),
        }
    }
}

impl From<DensityError> for DriftError {
    fn from(_e: DensityError) -> Self {
        DriftError::NoDensityTracker
    }
}

/// Frozen snapshot of a density tracker for drift detection.
#[derive(Debug, Clone)]
pub struct DriftBaseline {
    pub d: u32,
    /// Frozen baseline mean, shape `[d]`.
    pub mean: Tensor,
    /// Frozen baseline std (`sqrt(variance)`), shape `[d]`.
    pub std: Tensor,
    /// Number of observations the density tracker had at freeze time.
    pub n_at_freeze: u64,
    /// SHA-256 of the source density tracker's canonical bytes — links
    /// the baseline back to a specific tracker state.
    pub frozen_hash: [u8; 32],
}

impl DriftBaseline {
    /// Build a baseline by snapshotting a density tracker. The tracker
    /// must have at least 2 observations.
    pub fn from_density(density: &DensityTracker) -> Result<Self, DriftError> {
        if density.n < 2 {
            return Err(DriftError::InsufficientEvidence { n: density.n });
        }
        let dz = density.d as usize;
        let var = density.variance();
        let std: Vec<f64> = var.iter().map(|&v| v.sqrt()).collect();
        let std_t = Tensor::from_vec(std, &[dz]).expect("drift baseline std tensor");
        let mean_t = Tensor::from_vec(density.mean.to_vec(), &[dz])
            .expect("drift baseline mean tensor");
        let frozen_hash = density.state_hash();
        Ok(Self {
            d: density.d,
            mean: mean_t,
            std: std_t,
            n_at_freeze: density.n,
            frozen_hash,
        })
    }

    /// Drift score: per-dim z-shift L2-normalised.
    /// Returns `0.0` when current `n < 2` (no signal yet).
    pub fn drift_score(&self, current: &DensityTracker) -> Result<f64, DriftError> {
        if current.d != self.d {
            return Err(DriftError::DimMismatch {
                density_d: current.d,
                baseline_d: self.d,
            });
        }
        if current.n < 2 {
            return Ok(0.0);
        }
        let dz = self.d as usize;
        let mu_now = current.mean.to_vec();
        let mu_base = self.mean.to_vec();
        let std_base = self.std.to_vec();
        let mut acc = KahanAccumulatorF64::new();
        for i in 0..dz {
            let z = (mu_now[i] - mu_base[i]) / std_base[i].max(STD_FLOOR);
            acc.add(z * z);
        }
        Ok((acc.finalize() / dz as f64).sqrt())
    }

    /// Canonical big-endian byte encoding for hashing.
    /// Layout: `d u32 + n_at_freeze u64 + mean f64×d + std f64×d + frozen_hash`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let dz = self.d as usize;
        let mut out = Vec::with_capacity(4 + 8 + dz * 16 + 32);
        out.extend_from_slice(&self.d.to_be_bytes());
        out.extend_from_slice(&self.n_at_freeze.to_be_bytes());
        for x in self.mean.to_vec() {
            out.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        for x in self.std.to_vec() {
            out.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        out.extend_from_slice(&self.frozen_hash);
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

    fn fill(t: &mut DensityTracker, samples: &[(f64, f64)]) {
        let flat: Vec<f64> = samples.iter().flat_map(|&(a, b)| [a, b]).collect();
        t.observe_batch(&flat).unwrap();
    }

    #[test]
    fn freeze_requires_n_ge_two() {
        let t = DensityTracker::new(2);
        let err = DriftBaseline::from_density(&t).unwrap_err();
        assert!(matches!(err, DriftError::InsufficientEvidence { n: 0 }));
    }

    #[test]
    fn freeze_captures_mean_and_std() {
        let mut t = DensityTracker::new(1);
        t.observe_batch(&[1.0, 2.0, 3.0]).unwrap();
        // mean = 2, var = 1, std = 1
        let b = DriftBaseline::from_density(&t).unwrap();
        let mean = b.mean.to_vec();
        let std = b.std.to_vec();
        assert!((mean[0] - 2.0).abs() < 1e-12);
        assert!((std[0] - 1.0).abs() < 1e-12);
        assert_eq!(b.n_at_freeze, 3);
    }

    #[test]
    fn drift_zero_when_no_change() {
        let mut t = DensityTracker::new(1);
        t.observe_batch(&[1.0, 2.0, 3.0]).unwrap();
        let b = DriftBaseline::from_density(&t).unwrap();
        // Same tracker, same mean → drift = 0.
        let s = b.drift_score(&t).unwrap();
        assert!(s.abs() < 1e-12);
    }

    #[test]
    fn drift_increases_under_mean_shift() {
        let mut t = DensityTracker::new(1);
        t.observe_batch(&[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = DriftBaseline::from_density(&t).unwrap();
        // Continue observing samples shifted by +5.
        let mut t2 = t.clone();
        t2.observe_batch(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
        let s = b.drift_score(&t2).unwrap();
        assert!(s > 0.5, "drift score didn't rise: {s}");
    }

    #[test]
    fn drift_dim_mismatch_errs() {
        let mut t1 = DensityTracker::new(2);
        let mut t2 = DensityTracker::new(3);
        fill(&mut t1, &[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]);
        // hand-fill t2
        t2.observe_batch(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
            .unwrap();
        let b = DriftBaseline::from_density(&t1).unwrap();
        let err = b.drift_score(&t2).unwrap_err();
        assert!(matches!(err, DriftError::DimMismatch { .. }));
    }

    #[test]
    fn determinism_double_run() {
        let mk = || {
            let mut t = DensityTracker::new(2);
            fill(&mut t, &[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]);
            DriftBaseline::from_density(&t).unwrap()
        };
        let a = mk();
        let b = mk();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn canonical_bytes_size() {
        let mut t = DensityTracker::new(3);
        t.observe_batch(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let b = DriftBaseline::from_density(&t).unwrap();
        // 4 (d) + 8 (n) + 3*8 (mean) + 3*8 (std) + 32 (hash) = 92
        assert_eq!(b.canonical_bytes().len(), 92);
    }
}
