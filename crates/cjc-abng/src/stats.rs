//! Per-node sufficient statistics for ABNG.
//!
//! Phase 0.1 tracks the three Welford sufficient statistics for the running
//! mean and variance of a stream of `f64` observations:
//!
//! * `n_seen` — total observations (`u64`)
//! * `mean` — running arithmetic mean (`f64`, recurrence `mean += (x − mean) / n`)
//! * `m2` — Kahan-compensated sum of squared deviations (`KahanAccumulatorF64`)
//!
//! Variance is `m2 / (n − 1)` (sample variance) when `n ≥ 2`, else `0.0`.
//!
//! # Determinism
//!
//! For a fixed sample order, all updates produce bit-identical results across
//! runs and platforms:
//!
//! * The Welford `mean` recurrence uses only `+`, `-`, `/` on `f64` with no
//!   FMA, so it is bit-deterministic when the inputs arrive in a fixed order.
//! * `M2` accumulation goes through `KahanAccumulatorF64` from `cjc-repro`,
//!   which is documented as deterministic for a fixed addition order.
//!
//! Phase 0.1 is single-threaded; cross-thread determinism is the next phase's
//! problem (it'll need `BinnedAccumulator` in place of Kahan).
//!
//! # Canonical bytes
//!
//! [`NodeStats::canonical_bytes`] produces a fixed 24-byte big-endian
//! encoding suitable for hashing. The byte layout is part of the chain
//! hash contract — changing it breaks audit-log compatibility.

use cjc_repro::KahanAccumulatorF64;

/// Sufficient statistics for a stream of `f64` observations.
#[derive(Debug, Clone)]
pub struct NodeStats {
    /// Total number of observations applied via [`Self::observe`].
    pub n_seen: u64,
    /// Running arithmetic mean.
    pub mean: f64,
    /// Kahan-compensated running sum of squared deviations from the mean.
    pub m2: KahanAccumulatorF64,
}

impl NodeStats {
    /// Construct a zero-initialized [`NodeStats`].
    pub fn new() -> Self {
        Self {
            n_seen: 0,
            mean: 0.0,
            m2: KahanAccumulatorF64::new(),
        }
    }

    /// Apply one observation in Welford's streaming form.
    ///
    /// The update is `mean += (x − mean) / n` followed by a Kahan-compensated
    /// `M2 += δ · δ′`, where `δ = x − mean_prev` and `δ′ = x − mean_new`.
    /// This is the canonical online algorithm for variance and is bit-stable
    /// for a fixed sample order.
    pub fn observe(&mut self, value: f64) {
        self.n_seen += 1;
        let n = self.n_seen as f64;
        let delta = value - self.mean;
        self.mean += delta / n;
        let delta2 = value - self.mean;
        self.m2.add(delta * delta2);
    }

    /// Apply observations in slice order. Equivalent to calling [`Self::observe`]
    /// for each element; the explicit method exists so the dispatch layer can
    /// document tensor-order-as-update-order at the boundary.
    pub fn observe_slice(&mut self, values: &[f64]) {
        for &v in values {
            self.observe(v);
        }
    }

    /// Combine two streams of observations into `self` via Chan/Golub/
    /// LeVeque parallel Welford merge — mathematically equivalent to
    /// having observed both streams in (their original) order, but the
    /// merge itself only looks at the summary statistics, not the raw
    /// values.
    ///
    /// ```text
    ///   n_c    = n1 + n2
    ///   delta  = mean2 - mean1
    ///   mean_c = mean1 + delta * n2 / n_c
    ///   M2_c   = M2_1 + M2_2 + delta² * n1 * n2 / n_c
    /// ```
    ///
    /// Phase 0.4 Track B-2.2.6 — used by the graph layer's `force_merge`
    /// (and the policy-driven Merge fired from `decide_step`) to fold
    /// `absorbed`'s observation history into `into` instead of dropping
    /// it.
    pub fn combine(&mut self, other: &NodeStats) {
        if other.n_seen == 0 {
            return;
        }
        if self.n_seen == 0 {
            self.n_seen = other.n_seen;
            self.mean = other.mean;
            // Replace the Kahan accumulator with one whose finalized
            // value matches `other.m2.finalize()`. This preserves the
            // canonical_bytes contract (which serializes only the
            // finalized M2) without surfacing the compensation
            // register cross-instance.
            self.m2 = KahanAccumulatorF64::new();
            self.m2.add(other.m2.finalize());
            return;
        }
        let n1 = self.n_seen as f64;
        let n2 = other.n_seen as f64;
        let n_c = n1 + n2;
        let delta = other.mean - self.mean;
        let new_mean = self.mean + delta * (n2 / n_c);
        let m2_correction = delta * delta * (n1 * n2) / n_c;
        let m2_combined = self.m2.finalize() + other.m2.finalize() + m2_correction;
        // Reset accumulator to the combined finalized value (canonical
        // bytes only persist the finalized number, so this preserves
        // determinism through serialize→replay).
        self.m2 = KahanAccumulatorF64::new();
        self.m2.add(m2_combined);
        self.mean = new_mean;
        self.n_seen = self.n_seen.saturating_add(other.n_seen);
    }

    /// Sample variance (`M2 / (n − 1)`); returns `0.0` if `n < 2`.
    ///
    /// Phase 0.1 returns sample (Bessel-corrected) variance because the chess
    /// RL demo's value-head application is point estimation, not population
    /// modeling. Population variance (`M2 / n`) would be a one-line change in
    /// 0.2 if the workload changes.
    pub fn variance(&self) -> f64 {
        if self.n_seen < 2 {
            0.0
        } else {
            self.m2.finalize() / (self.n_seen - 1) as f64
        }
    }

    /// Canonical 24-byte encoding for hashing.
    ///
    /// Layout (all big-endian):
    /// ```text
    ///   [0..8]   n_seen          u64
    ///   [8..16]  mean.to_bits()  u64 (IEEE-754 bit pattern)
    ///   [16..24] m2.finalize()   u64 (IEEE-754 bit pattern)
    /// ```
    /// Using `to_bits()` instead of `to_le_bytes()` preserves signaling-NaN
    /// patterns; using big-endian keeps the canonical form platform-stable.
    pub fn canonical_bytes(&self) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[0..8].copy_from_slice(&self.n_seen.to_be_bytes());
        out[8..16].copy_from_slice(&self.mean.to_bits().to_be_bytes());
        out[16..24].copy_from_slice(&self.m2.finalize().to_bits().to_be_bytes());
        out
    }

    /// SHA-256 of [`canonical_bytes`](Self::canonical_bytes), for the audit
    /// chain.
    pub fn stats_hash(&self) -> [u8; 32] {
        cjc_snap::hash::sha256(&self.canonical_bytes())
    }
}

impl Default for NodeStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_stats_zero() {
        let s = NodeStats::new();
        assert_eq!(s.n_seen, 0);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.variance(), 0.0);
    }

    #[test]
    fn welford_matches_textbook_one_sample() {
        let mut s = NodeStats::new();
        s.observe(5.0);
        assert_eq!(s.n_seen, 1);
        assert_eq!(s.mean, 5.0);
        assert_eq!(s.variance(), 0.0); // n < 2
    }

    #[test]
    fn welford_matches_textbook_two_samples() {
        let mut s = NodeStats::new();
        s.observe(1.0);
        s.observe(3.0);
        // mean = 2.0; m2 = (1-1)*(1-2) + (3-2)*(3-2) ... let me trace:
        // step1: n=1, delta=1, mean=1, delta2=0, m2 += 1*0 = 0
        // step2: n=2, delta=2, mean=2, delta2=1, m2 += 2*1 = 2
        // var = 2 / (2-1) = 2.0
        assert_eq!(s.n_seen, 2);
        assert_eq!(s.mean, 2.0);
        assert_eq!(s.variance(), 2.0);
    }

    #[test]
    fn welford_matches_naive_on_simple_sequence() {
        let xs: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let mut s = NodeStats::new();
        s.observe_slice(&xs);

        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert_eq!(s.n_seen, 10);
        assert!((s.mean - mean).abs() < 1e-12);
        assert!((s.variance() - var).abs() < 1e-12);
    }

    #[test]
    fn canonical_bytes_size_24() {
        let s = NodeStats::new();
        assert_eq!(s.canonical_bytes().len(), 24);
    }

    #[test]
    fn stats_hash_changes_on_observe() {
        let mut s = NodeStats::new();
        let h0 = s.stats_hash();
        s.observe(1.0);
        let h1 = s.stats_hash();
        assert_ne!(h0, h1);
    }

    #[test]
    fn deterministic_double_run() {
        let xs: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.001).collect();
        let mut s1 = NodeStats::new();
        let mut s2 = NodeStats::new();
        s1.observe_slice(&xs);
        s2.observe_slice(&xs);
        assert_eq!(s1.canonical_bytes(), s2.canonical_bytes());
        assert_eq!(s1.stats_hash(), s2.stats_hash());
    }

    // ── Phase 0.4 Track B-2.2.6: parallel Welford merge ──────────────

    #[test]
    fn combine_matches_serial_observation() {
        // The combined stats of two streams must (within FP error)
        // equal the stats of having observed the concatenation.
        let xs1: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
        let xs2: Vec<f64> = (100..250).map(|i| (i as f64) * 0.01).collect();
        let mut s1 = NodeStats::new();
        s1.observe_slice(&xs1);
        let mut s2 = NodeStats::new();
        s2.observe_slice(&xs2);
        s1.combine(&s2);

        let mut serial = NodeStats::new();
        serial.observe_slice(&xs1);
        serial.observe_slice(&xs2);

        assert_eq!(s1.n_seen, serial.n_seen);
        assert!(
            (s1.mean - serial.mean).abs() < 1e-12,
            "combine mean {} vs serial mean {}",
            s1.mean,
            serial.mean
        );
        // Variance can drift slightly because the combine works on
        // finalized M2 (not the Kahan compensation register), but for
        // small streams the agreement should be tight.
        assert!(
            (s1.variance() - serial.variance()).abs() < 1e-9,
            "combine var {} vs serial var {}",
            s1.variance(),
            serial.variance()
        );
    }

    #[test]
    fn combine_with_empty_other_is_noop() {
        let mut s = NodeStats::new();
        s.observe_slice(&[1.0, 2.0, 3.0]);
        let pre = s.canonical_bytes();
        s.combine(&NodeStats::new());
        assert_eq!(s.canonical_bytes(), pre);
    }

    #[test]
    fn combine_with_empty_self_takes_other() {
        let mut self_ = NodeStats::new();
        let mut other = NodeStats::new();
        other.observe_slice(&[1.0, 2.0, 3.0]);
        self_.combine(&other);
        assert_eq!(self_.n_seen, 3);
        assert_eq!(self_.mean, 2.0);
    }

    #[test]
    fn combine_is_commutative_in_n_and_mean() {
        // n_seen and mean must be identical regardless of which side is
        // self vs other (variance can differ slightly because the
        // dropped Kahan compensation register is summed-in
        // non-symmetrically).
        let xs1: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let xs2: Vec<f64> = (50..120).map(|i| (i as f64) * 0.1).collect();
        let mut a1 = NodeStats::new();
        a1.observe_slice(&xs1);
        let mut a2 = NodeStats::new();
        a2.observe_slice(&xs2);
        let mut b1 = NodeStats::new();
        b1.observe_slice(&xs1);
        let mut b2 = NodeStats::new();
        b2.observe_slice(&xs2);

        a1.combine(&a2);
        b2.combine(&b1);
        assert_eq!(a1.n_seen, b2.n_seen);
        assert!((a1.mean - b2.mean).abs() < 1e-12);
    }
}
