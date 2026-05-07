//! Structural-decision policy thresholds — Phase 0.3d-3.
//!
//! `DecisionPolicy` is the graph-wide configuration that 0.3d-4's
//! `decide_step` engine consults to choose when to fire each
//! structural action (Grow, Split, Merge, Prune, Compress, Freeze).
//!
//! Phase 0.3d-3 ships **only the storage**. The thresholds are
//! installed via `abng_set_decision_policy` and read back via
//! `abng_action_count` / `abng_is_frozen` — but no policy-driven
//! mutations fire yet. Force-* builtins bypass the policy entirely.
//!
//! # Why a single `Tensor[11]` install API
//!
//! Following the [`set_codebook`](crate::AdaptiveBeliefGraph::set_codebook)
//! pattern: a 1-D Tensor of `f64` values in canonical order. The
//! field-order is documented here and locked in to the v7 snapshot
//! format. Eleven separate Float arguments would be unwieldy in
//! `.cjcl` source.
//!
//! # Canonical field order
//!
//! ```text
//!   [0]  H_grow              — route entropy threshold for Grow
//!   [1]  grow_min            — min samples_seen for Grow                (cast u64)
//!   [2]  split_min           — min samples_seen for Split               (cast u64)
//!   [3]  nll_split_gain      — held-out ΔNLL gain for Split
//!   [4]  impurity_min        — impurity decrease threshold for Split
//!   [5]  tau_merge           — Hamming distance threshold for Merge     (cast u8, ≤ 32)
//!   [6]  kl_merge            — posterior KL threshold for Merge
//!   [7]  prune_floor         — sample count below which Prune permitted (cast u64)
//!   [8]  prune_grace_epochs  — signature-stable epochs before Prune     (cast u64)
//!   [9]  tau_compress        — Hamming distance threshold for Compress  (cast u8, ≤ 32)
//!   [10] freeze_after        — signature-stable epochs before Freeze    (cast u64)
//! ```
//!
//! Canonical bytes: `11 × f64::to_bits().to_be_bytes()` = 88 bytes.

use cjc_repro::KahanAccumulatorF64;

/// Number of thresholds in a [`DecisionPolicy`]. Frozen — every snapshot
/// from v7 onward expects exactly this many bytes in the policy section.
pub const N_THRESHOLDS: usize = 11;

/// Total canonical-bytes length of a [`DecisionPolicy`]:
/// `N_THRESHOLDS * size_of::<f64>()` = 88 bytes.
pub const POLICY_BYTES_LEN: usize = N_THRESHOLDS * 8;

/// Errors specific to [`DecisionPolicy`] construction.
#[derive(Debug, PartialEq)]
pub enum PolicyError {
    /// Caller passed a Tensor whose 1-D length isn't [`N_THRESHOLDS`].
    WrongLength { got: usize, expected: usize },
    /// One or more thresholds is non-finite.
    NonFinite { index: usize, value: f64 },
    /// `tau_merge` or `tau_compress` is outside the valid Hamming-distance
    /// range `[0, 32]` (signature is 32 bytes wide).
    InvalidHamming { index: usize, value: f64 },
    /// A threshold whose semantics require non-negative values is negative.
    NegativeThreshold { index: usize, value: f64 },
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyError::WrongLength { got, expected } => write!(
                f,
                "abng decision policy: thresholds length {got} != expected {expected}"
            ),
            PolicyError::NonFinite { index, value } => write!(
                f,
                "abng decision policy: thresholds[{index}] = {value} is not finite"
            ),
            PolicyError::InvalidHamming { index, value } => write!(
                f,
                "abng decision policy: thresholds[{index}] = {value} must be in [0, 32]"
            ),
            PolicyError::NegativeThreshold { index, value } => write!(
                f,
                "abng decision policy: thresholds[{index}] = {value} must be non-negative"
            ),
        }
    }
}

/// Frozen graph-wide thresholds for 0.3d-4's structural-decision engine.
#[derive(Debug, Clone)]
pub struct DecisionPolicy {
    /// Raw f64 thresholds in canonical order (see module docs).
    pub thresholds: [f64; N_THRESHOLDS],
    /// SHA-256 of canonical bytes — embedded in the snapshot header so
    /// replay can verify the loaded policy hasn't been tampered with.
    pub policy_hash: [u8; 32],
}

impl DecisionPolicy {
    /// Construct from a slice of exactly `N_THRESHOLDS` f64 values.
    /// Validates finiteness, Hamming-range bounds (indices 5, 9), and
    /// non-negativity for the cast-to-u64 fields (indices 1, 2, 7, 8, 10).
    pub fn new(values: &[f64]) -> Result<Self, PolicyError> {
        if values.len() != N_THRESHOLDS {
            return Err(PolicyError::WrongLength {
                got: values.len(),
                expected: N_THRESHOLDS,
            });
        }
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(PolicyError::NonFinite { index: i, value: v });
            }
        }
        // Hamming-distance fields (tau_merge at 5, tau_compress at 9)
        // must fit in [0, 32].
        for &i in &[5usize, 9] {
            let v = values[i];
            if !(0.0..=32.0).contains(&v) {
                return Err(PolicyError::InvalidHamming { index: i, value: v });
            }
        }
        // Cast-to-u64 fields must be non-negative.
        for &i in &[1usize, 2, 7, 8, 10] {
            if values[i] < 0.0 {
                return Err(PolicyError::NegativeThreshold {
                    index: i,
                    value: values[i],
                });
            }
        }
        let mut thresholds = [0.0f64; N_THRESHOLDS];
        thresholds.copy_from_slice(values);
        let canonical = canonical_bytes_inner(&thresholds);
        let policy_hash = cjc_snap::hash::sha256(&canonical);
        Ok(Self {
            thresholds,
            policy_hash,
        })
    }

    /// Canonical big-endian byte encoding for hashing / snapshot.
    /// Layout: `f64::to_bits().to_be_bytes()` × `N_THRESHOLDS` = 88 bytes.
    pub fn canonical_bytes(&self) -> [u8; POLICY_BYTES_LEN] {
        canonical_bytes_inner(&self.thresholds)
    }

    /// SHA-256 of canonical bytes. Equal to `policy_hash` after a
    /// successful `new()`; recomputed for snapshot-replay verification.
    pub fn state_hash(&self) -> [u8; 32] {
        cjc_snap::hash::sha256(&self.canonical_bytes())
    }

    /// `H_grow` — route entropy threshold for Grow.
    pub fn h_grow(&self) -> f64 {
        self.thresholds[0]
    }

    /// `grow_min` — minimum `samples_seen` for Grow to fire.
    pub fn grow_min(&self) -> u64 {
        self.thresholds[1] as u64
    }

    /// `split_min` — minimum `samples_seen` for Split to fire.
    pub fn split_min(&self) -> u64 {
        self.thresholds[2] as u64
    }

    /// `nll_split_gain` — held-out ΔNLL gain threshold for Split.
    pub fn nll_split_gain(&self) -> f64 {
        self.thresholds[3]
    }

    /// `impurity_min` — impurity decrease threshold for Split.
    pub fn impurity_min(&self) -> f64 {
        self.thresholds[4]
    }

    /// `tau_merge` — Hamming-distance threshold for Merge (≤ 32).
    pub fn tau_merge(&self) -> u8 {
        self.thresholds[5] as u8
    }

    /// `kl_merge` — posterior-KL threshold for Merge.
    pub fn kl_merge(&self) -> f64 {
        self.thresholds[6]
    }

    /// `prune_floor` — `samples_seen` below which Prune is permitted.
    pub fn prune_floor(&self) -> u64 {
        self.thresholds[7] as u64
    }

    /// `prune_grace_epochs` — signature-stable epoch count before Prune fires.
    pub fn prune_grace_epochs(&self) -> u64 {
        self.thresholds[8] as u64
    }

    /// `tau_compress` — Hamming-distance threshold for Compress (≤ 32).
    pub fn tau_compress(&self) -> u8 {
        self.thresholds[9] as u8
    }

    /// `freeze_after` — signature-stable epoch count before Freeze fires.
    pub fn freeze_after(&self) -> u64 {
        self.thresholds[10] as u64
    }
}

/// Helper used by `new` and `canonical_bytes` so the encoding is
/// computed in exactly one place.
fn canonical_bytes_inner(thresholds: &[f64; N_THRESHOLDS]) -> [u8; POLICY_BYTES_LEN] {
    let mut out = [0u8; POLICY_BYTES_LEN];
    for (i, &v) in thresholds.iter().enumerate() {
        let bytes = v.to_bits().to_be_bytes();
        out[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }
    out
}

/// Sum the threshold values via Kahan compensation. Provided so that
/// future smoke-tests / canonical-bytes invariants can verify the
/// thresholds are arithmetically well-defined without leaking
/// non-deterministic accumulation order. Not used by the encoder.
#[allow(dead_code)]
fn deterministic_threshold_sum(thresholds: &[f64; N_THRESHOLDS]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for &v in thresholds {
        acc.add(v);
    }
    acc.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_thresholds() -> [f64; N_THRESHOLDS] {
        // Reasonable defaults — every check passes.
        [
            0.5, // H_grow
            64.0, // grow_min
            128.0, // split_min
            0.05, // nll_split_gain
            0.02, // impurity_min
            4.0, // tau_merge (Hamming)
            0.1, // kl_merge
            32.0, // prune_floor
            10.0, // prune_grace_epochs
            8.0, // tau_compress (Hamming)
            20.0, // freeze_after
        ]
    }

    #[test]
    fn n_thresholds_locked() {
        assert_eq!(N_THRESHOLDS, 11);
        assert_eq!(POLICY_BYTES_LEN, 88);
    }

    #[test]
    fn new_happy_path() {
        let p = DecisionPolicy::new(&ok_thresholds()).unwrap();
        assert_eq!(p.h_grow(), 0.5);
        assert_eq!(p.grow_min(), 64);
        assert_eq!(p.split_min(), 128);
        assert_eq!(p.tau_merge(), 4);
        assert_eq!(p.tau_compress(), 8);
        assert_eq!(p.freeze_after(), 20);
    }

    #[test]
    fn new_wrong_length_errs() {
        let err = DecisionPolicy::new(&[0.0; 10]).unwrap_err();
        assert_eq!(
            err,
            PolicyError::WrongLength {
                got: 10,
                expected: 11
            }
        );
    }

    #[test]
    fn new_non_finite_errs() {
        let mut t = ok_thresholds();
        t[3] = f64::NAN;
        let err = DecisionPolicy::new(&t).unwrap_err();
        assert!(matches!(err, PolicyError::NonFinite { index: 3, .. }));
        t[3] = f64::INFINITY;
        let err = DecisionPolicy::new(&t).unwrap_err();
        assert!(matches!(err, PolicyError::NonFinite { index: 3, .. }));
    }

    #[test]
    fn new_invalid_hamming_errs() {
        let mut t = ok_thresholds();
        t[5] = 33.0; // tau_merge > 32
        let err = DecisionPolicy::new(&t).unwrap_err();
        assert!(matches!(err, PolicyError::InvalidHamming { index: 5, .. }));
        let mut t = ok_thresholds();
        t[9] = -1.0; // tau_compress < 0
        let err = DecisionPolicy::new(&t).unwrap_err();
        assert!(matches!(err, PolicyError::InvalidHamming { index: 9, .. }));
    }

    #[test]
    fn new_negative_cast_field_errs() {
        let mut t = ok_thresholds();
        t[1] = -1.0; // grow_min must be non-negative
        let err = DecisionPolicy::new(&t).unwrap_err();
        assert!(matches!(err, PolicyError::NegativeThreshold { index: 1, .. }));
    }

    #[test]
    fn canonical_bytes_size() {
        let p = DecisionPolicy::new(&ok_thresholds()).unwrap();
        assert_eq!(p.canonical_bytes().len(), POLICY_BYTES_LEN);
        assert_eq!(POLICY_BYTES_LEN, 88);
    }

    #[test]
    fn state_hash_matches_policy_hash() {
        let p = DecisionPolicy::new(&ok_thresholds()).unwrap();
        assert_eq!(p.state_hash(), p.policy_hash);
    }

    #[test]
    fn determinism_double_run() {
        let a = DecisionPolicy::new(&ok_thresholds()).unwrap();
        let b = DecisionPolicy::new(&ok_thresholds()).unwrap();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_eq!(a.policy_hash, b.policy_hash);
    }

    #[test]
    fn distinct_thresholds_distinct_hash() {
        let a = DecisionPolicy::new(&ok_thresholds()).unwrap();
        let mut t = ok_thresholds();
        t[0] += 1e-10;
        let b = DecisionPolicy::new(&t).unwrap();
        assert_ne!(a.policy_hash, b.policy_hash);
    }

    #[test]
    fn canonical_bytes_layout_first_field() {
        // Field 0 (H_grow) lands in bytes [0..8] big-endian.
        let mut t = [0.0; N_THRESHOLDS];
        t[0] = 1.0;
        let p = DecisionPolicy::new(&t).unwrap();
        let bytes = p.canonical_bytes();
        let expected = 1.0_f64.to_bits().to_be_bytes();
        assert_eq!(&bytes[0..8], &expected);
        // Remaining 80 bytes are zero (since the rest are 0.0 and
        // 0.0_f64.to_bits() == 0).
        assert_eq!(&bytes[8..], &[0u8; 80]);
    }
}
