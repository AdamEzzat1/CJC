//! Per-node calibration bins — Phase 0.3c.
//!
//! Standard reliability-diagram math: bin predictions by their predicted
//! probability `p ∈ [0, 1]`, count actual successes per bin, then read
//! ECE as the weighted gap between predicted and observed accuracy.
//!
//! ```text
//!   ECE = Σ_b (counts[b] / N) · |correct_counts[b]/counts[b] − conf_sum[b]/counts[b]|
//! ```
//!
//! Empty bins contribute zero. Phase 0.3d's `Maturity.calibration_stable`
//! will require minimum bin populations before declaring stable; 0.3c
//! just exposes the raw ECE.
//!
//! # Determinism
//!
//! `conf_sum` is stored as `f64::to_bits()` per bin to keep the
//! canonical-bytes encoding bit-deterministic. Updates use
//! `KahanAccumulatorF64` internally then `finalize()` to a stored
//! `f64`; the running compensation register is *not* persisted — it
//! lives only inside the update call. This trades one Kahan step per
//! update for a consistent serialisation contract.

use cjc_repro::KahanAccumulatorF64;

const MIN_BINS: u8 = 2;
const MAX_BINS: u8 = 100;

/// Errors specific to the calibration subsystem.
#[derive(Debug, PartialEq)]
pub enum CalibrationError {
    /// `set_calibration` was called twice on the same graph.
    AlreadyFrozen,
    /// `set_calibration` was called on a graph that already has child nodes.
    NotEmptyGraph { n_nodes: u32 },
    /// `n_bins` is outside `[2, 100]`.
    InvalidNumBins(u8),
    /// A calibration op was called on a node without an installed bin set.
    NoCalibration,
    /// `predicted_prob` was outside `[0.0, 1.0]` or non-finite.
    InvalidProbability(f64),
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationError::AlreadyFrozen => write!(f, "abng calibration: already frozen"),
            CalibrationError::NotEmptyGraph { n_nodes } => write!(
                f,
                "abng calibration: must be installed before any add_node \
                 (graph already has {n_nodes} nodes)"
            ),
            CalibrationError::InvalidNumBins(n) => write!(
                f,
                "abng calibration: n_bins {n} must be in [2, 100]"
            ),
            CalibrationError::NoCalibration => write!(f, "abng calibration: no bins installed"),
            CalibrationError::InvalidProbability(p) => write!(
                f,
                "abng calibration: predicted_prob {p} must be in [0, 1] and finite"
            ),
        }
    }
}

/// Per-node 15-bin reliability diagram.
#[derive(Debug, Clone)]
pub struct CalibrationBins {
    pub n_bins: u8,
    pub counts: Vec<u32>,
    pub correct_counts: Vec<u32>,
    /// `conf_sum[b]` stored as `f64::to_bits()`; finalize via
    /// `f64::from_bits` when reading.
    pub conf_sum_bits: Vec<u64>,
}

impl CalibrationBins {
    /// Construct a fresh bin set with `n_bins` buckets, all empty.
    pub fn new(n_bins: u8) -> Result<Self, CalibrationError> {
        if !(MIN_BINS..=MAX_BINS).contains(&n_bins) {
            return Err(CalibrationError::InvalidNumBins(n_bins));
        }
        let nb = n_bins as usize;
        Ok(Self {
            n_bins,
            counts: vec![0; nb],
            correct_counts: vec![0; nb],
            conf_sum_bits: vec![0u64; nb],
        })
    }

    /// Record one prediction: `predicted_prob` is the model's confidence
    /// in its top class (or the regression-success probability), and
    /// `was_correct` is the resolution.
    pub fn observe(
        &mut self,
        predicted_prob: f64,
        was_correct: bool,
    ) -> Result<(), CalibrationError> {
        if !predicted_prob.is_finite() || !(0.0..=1.0).contains(&predicted_prob) {
            return Err(CalibrationError::InvalidProbability(predicted_prob));
        }
        let nb = self.n_bins as usize;
        // bin = floor(p * n_bins), clamped to n_bins - 1 so p == 1.0 lands
        // in the top bin instead of going out of range.
        let mut bin = (predicted_prob * nb as f64).floor() as usize;
        if bin >= nb {
            bin = nb - 1;
        }
        self.counts[bin] = self.counts[bin].saturating_add(1);
        if was_correct {
            self.correct_counts[bin] = self.correct_counts[bin].saturating_add(1);
        }
        // Kahan-add into the running sum for this bin.
        let prev = f64::from_bits(self.conf_sum_bits[bin]);
        let mut acc = KahanAccumulatorF64::new();
        acc.add(prev);
        acc.add(predicted_prob);
        self.conf_sum_bits[bin] = acc.finalize().to_bits();
        Ok(())
    }

    /// Total observations.
    pub fn n_seen(&self) -> u64 {
        self.counts.iter().map(|&c| c as u64).sum()
    }

    /// Expected Calibration Error — weighted sum of `|accuracy − confidence|`
    /// per non-empty bin.
    pub fn ece(&self) -> f64 {
        let n_total = self.n_seen();
        if n_total == 0 {
            return 0.0;
        }
        let n_total_f = n_total as f64;
        let mut acc = KahanAccumulatorF64::new();
        for b in 0..(self.n_bins as usize) {
            let c = self.counts[b];
            if c == 0 {
                continue;
            }
            let c_f = c as f64;
            let acc_b = self.correct_counts[b] as f64 / c_f;
            let conf_b = f64::from_bits(self.conf_sum_bits[b]) / c_f;
            let gap = (acc_b - conf_b).abs();
            acc.add((c_f / n_total_f) * gap);
        }
        acc.finalize()
    }

    /// Canonical big-endian byte encoding for hashing.
    /// Layout: `n_bins u8 + counts u32×n_bins + correct_counts u32×n_bins
    /// + conf_sum_bits u64×n_bins`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let nb = self.n_bins as usize;
        let mut out = Vec::with_capacity(1 + nb * 16);
        out.push(self.n_bins);
        for &c in &self.counts {
            out.extend_from_slice(&c.to_be_bytes());
        }
        for &c in &self.correct_counts {
            out.extend_from_slice(&c.to_be_bytes());
        }
        for &cs in &self.conf_sum_bits {
            out.extend_from_slice(&cs.to_be_bytes());
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
    fn rejects_bad_n_bins() {
        assert_eq!(
            CalibrationBins::new(0).unwrap_err(),
            CalibrationError::InvalidNumBins(0)
        );
        assert_eq!(
            CalibrationBins::new(1).unwrap_err(),
            CalibrationError::InvalidNumBins(1)
        );
        assert_eq!(
            CalibrationBins::new(101).unwrap_err(),
            CalibrationError::InvalidNumBins(101)
        );
    }

    #[test]
    fn empty_bins_zero_ece() {
        let c = CalibrationBins::new(15).unwrap();
        assert_eq!(c.ece(), 0.0);
        assert_eq!(c.n_seen(), 0);
    }

    #[test]
    fn perfect_calibration_zero_ece() {
        let mut c = CalibrationBins::new(10).unwrap();
        // Predict 0.5 for every sample; half are correct.
        for i in 0..100 {
            c.observe(0.5, i % 2 == 0).unwrap();
        }
        assert!(c.ece() < 1e-9, "ECE should be ~0, got {}", c.ece());
    }

    #[test]
    fn miscalibrated_high_ece() {
        let mut c = CalibrationBins::new(10).unwrap();
        // Predict 0.9 for every sample; only 10% correct → big gap.
        for i in 0..100 {
            c.observe(0.9, i % 10 == 0).unwrap();
        }
        let ece = c.ece();
        assert!(ece > 0.7, "ECE should be ≈0.8, got {ece}");
    }

    #[test]
    fn observe_invalid_probability_errs() {
        let mut c = CalibrationBins::new(10).unwrap();
        assert!(matches!(
            c.observe(-0.1, true).unwrap_err(),
            CalibrationError::InvalidProbability(_)
        ));
        assert!(matches!(
            c.observe(1.1, true).unwrap_err(),
            CalibrationError::InvalidProbability(_)
        ));
        assert!(matches!(
            c.observe(f64::NAN, true).unwrap_err(),
            CalibrationError::InvalidProbability(_)
        ));
    }

    #[test]
    fn boundary_probabilities_land_correctly() {
        let mut c = CalibrationBins::new(10).unwrap();
        // p=0.0 → bin 0
        c.observe(0.0, true).unwrap();
        // p=1.0 → top bin (clamped from 10 to 9)
        c.observe(1.0, true).unwrap();
        assert_eq!(c.counts[0], 1);
        assert_eq!(c.counts[9], 1);
    }

    #[test]
    fn n_seen_accumulates() {
        let mut c = CalibrationBins::new(15).unwrap();
        for _ in 0..7 {
            c.observe(0.5, true).unwrap();
        }
        assert_eq!(c.n_seen(), 7);
    }

    #[test]
    fn determinism_double_run() {
        let mk = || {
            let mut c = CalibrationBins::new(15).unwrap();
            for i in 0..50 {
                let p = (i as f64) * 0.02;
                c.observe(p, i % 3 == 0).unwrap();
            }
            c
        };
        let a = mk();
        let b = mk();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn canonical_bytes_size() {
        let c = CalibrationBins::new(15).unwrap();
        // 1 (n_bins) + 15*4 (counts) + 15*4 (correct) + 15*8 (conf) = 241
        assert_eq!(c.canonical_bytes().len(), 241);
    }

    #[test]
    fn state_hash_changes_after_observe() {
        let mut c = CalibrationBins::new(15).unwrap();
        let h0 = c.state_hash();
        c.observe(0.5, true).unwrap();
        let h1 = c.state_hash();
        assert_ne!(h0, h1);
    }
}
