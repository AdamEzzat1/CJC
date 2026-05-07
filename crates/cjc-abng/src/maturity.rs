//! Per-node maturity summary — Phase 0.3d-1/4 (lazy / read-only).
//!
//! `Maturity` is the structural-decision policy's view of how much
//! evidence a node has accumulated and whether the evidence is stable
//! enough to be acted upon.
//!
//! ## Phase 0.3d-1 (foundation, lazy)
//! Stability flags were stubbed `false`; `trust_level` derived from
//! `samples_seen` alone.
//!
//! ## Phase 0.3d-4 (real flags)
//! `calibration_stable` and `uncertainty_stable` now read live evidence
//! against simple thresholds:
//!
//! * `calibration_stable` ← `node.calibration.is_some() && ece < ECE_STABILITY_MAX`
//! * `uncertainty_stable` ← `node.blr.is_some() && samples_seen ≥ UNCERTAINTY_STABLE_MIN_SAMPLES
//!     && signature_stable_calls ≥ 1`
//!
//! These thresholds are deliberately single-window (vs the prompt's
//! 3-window design) — the windowed refinement is deferred to Phase 0.4
//! alongside CLI tooling. The single-window form is sufficient to
//! drive `decide_step`'s auto-capture of `expected_epistemic` and the
//! evidence-gated triggers.
//!
//! # Layout (locked-in for the v6 snapshot bump in 0.3d-3)
//!
//! Canonical bytes are 11 bytes, big-endian:
//!
//! ```text
//!   samples_seen           u64 BE   (8)
//!   calibration_stable     u8       (1)   0x00 = false, 0x01 = true
//!   uncertainty_stable     u8       (1)   0x00 = false, 0x01 = true
//!   trust_level            u8       (1)   0..=4
//! ```
//!
//! The wire shape is fixed now so 0.3d-3 can promote `Maturity` to
//! persistent state without a second format break.

use crate::node::AdaptiveBeliefNode;

/// Highest trust level a node can attain. Frozen at `4` — adjustments
/// to the level *meaning* in later sub-steps must not change the
/// numeric range or the wire-byte size.
pub const MAX_TRUST_LEVEL: u8 = 4;

/// Number of observations after which `trust_level` reaches its cap of
/// `MAX_TRUST_LEVEL`. Picked at `MAX_TRUST_LEVEL × 64 = 256` so each
/// trust step requires 64 fresh samples — deliberately matches the
/// `min_required_samples = 64` default the policy will use for split.
pub const TRUST_SAMPLE_STEP: u64 = 64;

/// Phase 0.3d-4 — Expected Calibration Error threshold below which
/// `calibration_stable` flips to `true`. Single-window threshold;
/// Phase 0.4 will replace with a 3-window |ΔECE| comparison.
pub const ECE_STABILITY_MAX: f64 = 0.05;

/// Phase 0.3d-4 — minimum `samples_seen` before `uncertainty_stable`
/// can flip to `true`. Conservative default (100) — typically enough
/// to converge a small BLR posterior.
pub const UNCERTAINTY_STABLE_MIN_SAMPLES: u64 = 100;

/// Per-node maturity summary. Recomputed on every read in 0.3d-1; gains
/// persistent windowing state in 0.3d-3/4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maturity {
    /// Number of observations applied to the node's
    /// [`NodeStats`](crate::stats::NodeStats). Mirrors `node.stats.n_seen`.
    pub samples_seen: u64,
    /// `true` once calibration ECE has been stable across the policy's
    /// recent windows. **Always `false` in Phase 0.3d-1** — the
    /// windowing logic lands with `decide_step` in sub-step 0.3d-4.
    pub calibration_stable: bool,
    /// `true` once epistemic σ has been stable across the policy's
    /// recent windows. **Always `false` in Phase 0.3d-1.**
    pub uncertainty_stable: bool,
    /// 0..=`MAX_TRUST_LEVEL`. Monotonic in `samples_seen` modulo the
    /// 0.3d-4 stability requirement for levels ≥ 3.
    pub trust_level: u8,
}

impl Maturity {
    /// Compute a fresh maturity snapshot from a node's current state.
    /// Phase 0.3d-4 flips the stability stubs to real thresholds:
    /// `calibration_stable` reads ECE against [`ECE_STABILITY_MAX`];
    /// `uncertainty_stable` requires BLR installed,
    /// `samples_seen ≥ UNCERTAINTY_STABLE_MIN_SAMPLES`, and at least one
    /// `decide_step` call observing an unchanged signature.
    pub fn from_node(node: &AdaptiveBeliefNode) -> Self {
        let samples_seen = node.stats.n_seen;
        let calibration_stable = node
            .calibration
            .as_ref()
            .map(|c| c.ece() < ECE_STABILITY_MAX)
            .unwrap_or(false);
        let uncertainty_stable = node.blr.is_some()
            && samples_seen >= UNCERTAINTY_STABLE_MIN_SAMPLES
            && node.signature_stable_calls >= 1;
        let trust_level = trust_from_samples(samples_seen);
        Self {
            samples_seen,
            calibration_stable,
            uncertainty_stable,
            trust_level,
        }
    }

    /// Canonical big-endian byte encoding for hashing / future snapshot.
    /// Layout: see module docs.
    pub fn canonical_bytes(&self) -> [u8; 11] {
        let mut out = [0u8; 11];
        out[..8].copy_from_slice(&self.samples_seen.to_be_bytes());
        out[8] = self.calibration_stable as u8;
        out[9] = self.uncertainty_stable as u8;
        out[10] = self.trust_level;
        out
    }
}

/// Map a sample count to a 0..=`MAX_TRUST_LEVEL` step. Saturating so the
/// returned `u8` never exceeds the cap.
fn trust_from_samples(samples_seen: u64) -> u8 {
    let step = samples_seen / TRUST_SAMPLE_STEP;
    if step >= MAX_TRUST_LEVEL as u64 {
        MAX_TRUST_LEVEL
    } else {
        step as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AdaptiveBeliefGraph;

    #[test]
    fn from_node_zero_samples() {
        let g = AdaptiveBeliefGraph::new(7);
        let m = Maturity::from_node(&g.nodes[0]);
        assert_eq!(m.samples_seen, 0);
        assert!(!m.calibration_stable);
        assert!(!m.uncertainty_stable);
        assert_eq!(m.trust_level, 0);
    }

    #[test]
    fn trust_level_climbs_with_samples() {
        // 0 → 0, 63 → 0, 64 → 1, 127 → 1, 128 → 2, 256 → 4 (cap).
        assert_eq!(trust_from_samples(0), 0);
        assert_eq!(trust_from_samples(63), 0);
        assert_eq!(trust_from_samples(64), 1);
        assert_eq!(trust_from_samples(127), 1);
        assert_eq!(trust_from_samples(128), 2);
        assert_eq!(trust_from_samples(192), 3);
        assert_eq!(trust_from_samples(256), 4);
        // Saturates at the cap.
        assert_eq!(trust_from_samples(1_000_000), MAX_TRUST_LEVEL);
    }

    #[test]
    fn trust_level_via_observe() {
        let mut g = AdaptiveBeliefGraph::new(0);
        for _ in 0..64 {
            g.observe(0, 1.0).unwrap();
        }
        let m = Maturity::from_node(&g.nodes[0]);
        assert_eq!(m.samples_seen, 64);
        assert_eq!(m.trust_level, 1);
    }

    #[test]
    fn cal_unc_flags_false_without_subsystems() {
        // Even after lots of evidence, both flags stay false unless
        // calibration / BLR are installed. This is the
        // "no-subsystem-no-claim" invariant — a graph that hasn't
        // installed an evidence channel cannot declare stability on it.
        let mut g = AdaptiveBeliefGraph::new(0);
        for i in 0..200 {
            g.observe(0, i as f64).unwrap();
        }
        let m = Maturity::from_node(&g.nodes[0]);
        assert!(!m.calibration_stable);
        assert!(!m.uncertainty_stable);
    }

    #[test]
    fn calibration_stable_flips_when_ece_low() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_calibration(15).unwrap();
        // Calibrate with predictions exactly matching outcomes →
        // ECE ≈ 0 → below ECE_STABILITY_MAX (0.05).
        for i in 0..50 {
            g.calibration_observe(0, 0.5, i % 2 == 0).unwrap();
        }
        let m = Maturity::from_node(&g.nodes[0]);
        assert!(m.calibration_stable, "ECE-low should flip flag true");
    }

    #[test]
    fn calibration_unstable_when_ece_high() {
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_calibration(15).unwrap();
        // Predict 0.9 always but only 10% correct → ECE ≈ 0.8.
        for i in 0..50 {
            g.calibration_observe(0, 0.9, i % 10 == 0).unwrap();
        }
        let m = Maturity::from_node(&g.nodes[0]);
        assert!(!m.calibration_stable);
    }

    #[test]
    fn uncertainty_stable_requires_signature_stability_call() {
        use cjc_ad::pinn::Activation;
        let mut g = AdaptiveBeliefGraph::new(0);
        g.set_leaf_head(2, vec![], 1, Activation::None).unwrap();
        g.set_blr_prior(1.0, 1.5, 1.0).unwrap();
        for _ in 0..150 {
            g.observe(0, 1.0).unwrap();
        }
        // signature_stable_calls hasn't been advanced yet → flag false.
        let m_before = Maturity::from_node(&g.nodes[0]);
        assert!(!m_before.uncertainty_stable);
        // Manually bump (decide_step would do this normally).
        g.nodes[0].signature_stable_calls = 1;
        let m_after = Maturity::from_node(&g.nodes[0]);
        assert!(m_after.uncertainty_stable);
    }

    #[test]
    fn canonical_bytes_size() {
        let m = Maturity {
            samples_seen: 0,
            calibration_stable: false,
            uncertainty_stable: false,
            trust_level: 0,
        };
        assert_eq!(m.canonical_bytes().len(), 11);
    }

    #[test]
    fn canonical_bytes_layout() {
        let m = Maturity {
            samples_seen: 0x0102_0304_0506_0708,
            calibration_stable: true,
            uncertainty_stable: false,
            trust_level: 3,
        };
        let bytes = m.canonical_bytes();
        assert_eq!(
            &bytes,
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x01, 0x00, 0x03]
        );
    }

    #[test]
    fn determinism_double_run() {
        let mk = || {
            let mut g = AdaptiveBeliefGraph::new(42);
            for i in 0..50 {
                g.observe(0, i as f64).unwrap();
            }
            Maturity::from_node(&g.nodes[0]).canonical_bytes()
        };
        assert_eq!(mk(), mk());
    }
}
