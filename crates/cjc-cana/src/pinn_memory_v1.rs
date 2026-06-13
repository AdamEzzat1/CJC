//! Phase F1 — the TRAINED memory head.
//!
//! ## Lineage (third run of the information-gap pattern)
//!
//! - PINN v2 (thermal): label was fine, FEATURES were type-blind →
//!   `TypeMix` closed the gap → R²(test) −0.05 → +0.96.
//! - Phase F0 (memory, label side): the recorded label was blind to
//!   Rc-buffer allocation → `alloc_bytes_in_window` closed it →
//!   corpus label std 0.0009 → 0.1083.
//! - Phase F1 (memory, feature side — THIS module): with the label
//!   fixed, the sanity pass measured R²(train) 0.77 / R²(test) 0.048 —
//!   the textbook information-gap signature, because `alloc_sites`
//!   counts a 2-element and a 774-element literal identically. The
//!   `lit_elem_slots` static counter + `creation_alloc_bytes_estimate`
//!   query field carry the missing volume signal; this head consumes
//!   them.
//!
//! Weights are fit OFFLINE by `bench/cana_train_pinn -- train-memory`
//! (deterministic closed-form ridge — no RNG, bit-reproducible) and
//! persisted as a CPB2 bundle (`cjc-cana-compress::memory_bundle`).
//! Compilation only ever LOADS weights. The head ships SHADOW-FIRST:
//! it is not attached to any cost model until its shadow gate says
//! PROMOTE (the same contract every head before it honored).
//!
//! ## Feature basis
//!
//! Eight features, all derivable from a [`PhysicalCostQuery`] (one
//! definition shared by trainer and predictor):
//!
//! | idx | feature |
//! |-----|---------|
//! | 0 | `ln(1 + flops_estimate)` |
//! | 1 | `ln(1 + bytes_read_estimate)` |
//! | 2 | `ln(1 + bytes_written_estimate)` |
//! | 3 | `ln(1 + allocation_bytes_estimate)` (site-count proxy — kept so the head can learn what sites alone explain) |
//! | 4 | `ln(1 + working_set_bytes_estimate)` |
//! | 5 | `ln(1 + float_ops_estimate)` |
//! | 6 | `ln(1 + creation_alloc_bytes_estimate)` — the Phase F1 volume signal |
//! | 7 | creation-alloc density `min(creation_alloc / (8·flops + 1), 1)` — churn per unit of work, loop-amplification-invariant |
//!
//! ## Determinism
//!
//! `predict_memory` is a pure function of `(weights, query)`:
//! fixed-order standardize → dot with named intermediates (no FMA) →
//! clamp to `[0, 1]` (the label range).

use crate::physical_cost::PhysicalCostQuery;

/// Stable model identifier; flows into report hashes if/when a trained
/// head is attached post-PROMOTE.
pub const PINN_MEMORY_V1_MODEL_ID: &str = "pinn_memory_v1";

/// Monotonic model version; bump on any change to the feature basis,
/// standardization scheme, or training recipe.
pub const PINN_MEMORY_V1_MODEL_VERSION: u32 = 1;

/// Number of features in the memory-v1 basis (see module docs table).
pub const PINN_MEMORY_V1_FEATURE_COUNT: usize = 8;

/// Trained linear memory head: standardization parameters + ridge
/// coefficients, fit offline by `bench/cana_train_pinn`.
#[derive(Debug, Clone, PartialEq)]
pub struct PinnMemoryV1 {
    /// Per-feature training means (z-score centering).
    pub feature_means: [f64; PINN_MEMORY_V1_FEATURE_COUNT],
    /// Per-feature training standard deviations. Zero-variance features
    /// are stored as 1.0 by the trainer.
    pub feature_stds: [f64; PINN_MEMORY_V1_FEATURE_COUNT],
    /// Ridge coefficients over the standardized features.
    pub coefficients: [f64; PINN_MEMORY_V1_FEATURE_COUNT],
    /// Intercept term.
    pub intercept: f64,
}

impl PinnMemoryV1 {
    /// `true` iff every parameter is finite and every std is strictly
    /// positive. Invalid heads must never be attached — loaders reject
    /// them at the boundary.
    pub fn is_valid(&self) -> bool {
        let finite = self
            .feature_means
            .iter()
            .chain(self.feature_stds.iter())
            .chain(self.coefficients.iter())
            .all(|v| v.is_finite())
            && self.intercept.is_finite();
        let stds_positive = self.feature_stds.iter().all(|s| *s > 0.0);
        finite && stds_positive
    }

    /// Predict memory pressure in `[0, 1]` for one workload query.
    ///
    /// Fixed-order standardize → dot → clamp; no FMA (each product is
    /// bound to a named intermediate before any addition).
    pub fn predict_memory(&self, query: &PhysicalCostQuery<'_>) -> f64 {
        let x = memory_features_from_query(query);
        let mut acc = self.intercept;
        for i in 0..PINN_MEMORY_V1_FEATURE_COUNT {
            let centered = x[i] - self.feature_means[i];
            let scaled = centered / self.feature_stds[i];
            let term = self.coefficients[i] * scaled;
            acc += term;
        }
        if acc.is_nan() {
            return 0.0;
        }
        acc.clamp(0.0, 1.0)
    }
}

/// Build the memory-v1 feature vector from a workload query — the
/// single definition both the trainer (row sums lifted into a synthetic
/// query) and the predictor use.
pub fn memory_features_from_query(
    q: &PhysicalCostQuery<'_>,
) -> [f64; PINN_MEMORY_V1_FEATURE_COUNT] {
    // Allocation churn per unit of work. The 8·flops denominator puts
    // both sides on a bytes scale (8 B per priced expr); capped at 1.0
    // because the recorded label saturates at its capacity clamp —
    // density beyond that carries no label information (the magnitude
    // survives uncapped in feature 6). The zero-flops case is guarded
    // explicitly (not via a `+1` epsilon) so the ratio stays EXACTLY
    // loop-amplification-invariant: numerator and denominator both
    // scale by `loop_amp`, which cancels only without an additive term.
    let density = if q.flops_estimate == 0 {
        if q.creation_alloc_bytes_estimate == 0 {
            0.0
        } else {
            1.0
        }
    } else {
        (q.creation_alloc_bytes_estimate as f64 / (8.0 * q.flops_estimate as f64)).min(1.0)
    };
    [
        log1p_u64(q.flops_estimate),
        log1p_u64(q.bytes_read_estimate),
        log1p_u64(q.bytes_written_estimate),
        log1p_u64(q.allocation_bytes_estimate),
        log1p_u64(q.working_set_bytes_estimate),
        log1p_u64(q.float_ops_estimate),
        log1p_u64(q.creation_alloc_bytes_estimate),
        density,
    ]
}

fn log1p_u64(v: u64) -> f64 {
    (v as f64).ln_1p()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn query_with(flops: u64, creation: u64) -> PhysicalCostQuery<'static> {
        PhysicalCostQuery {
            function_name: "f",
            strategy_id: "dce",
            flops_estimate: flops,
            bytes_read_estimate: flops.saturating_mul(8),
            bytes_written_estimate: 0,
            allocation_bytes_estimate: 0,
            working_set_bytes_estimate: 0,
            thread_count: 1,
            batch_size: 1,
            compression_overhead_bytes: 0,
            float_ops_estimate: 0,
            creation_alloc_bytes_estimate: creation,
        }
    }

    /// A head that passes the creation-alloc density through unchanged.
    fn density_passthrough() -> PinnMemoryV1 {
        PinnMemoryV1 {
            feature_means: [0.0; PINN_MEMORY_V1_FEATURE_COUNT],
            feature_stds: [1.0; PINN_MEMORY_V1_FEATURE_COUNT],
            coefficients: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            intercept: 0.0,
        }
    }

    #[test]
    fn density_is_loop_amplification_invariant() {
        // creation and flops both scale with loop_amp; the ratio holds.
        let a = memory_features_from_query(&query_with(100, 400));
        let b = memory_features_from_query(&query_with(1000, 4000));
        assert!((a[7] - b[7]).abs() < 1e-12);
    }

    #[test]
    fn zero_flops_density_is_finite_and_capped() {
        let x = memory_features_from_query(&query_with(0, 0));
        assert_eq!(x[7], 0.0);
        let x = memory_features_from_query(&query_with(0, u64::MAX));
        assert_eq!(x[7], 1.0, "cap must hold even with a unit denominator");
    }

    #[test]
    fn volume_signal_separates_slot_counts_at_equal_sites() {
        // The exact blindness this head exists to fix: same work, very
        // different creation volume must produce different feature 6.
        let small = memory_features_from_query(&query_with(1000, 64));
        let large = memory_features_from_query(&query_with(1000, 64 * 1024));
        assert!(large[6] > small[6] + 5.0);
    }

    #[test]
    fn passthrough_head_predicts_density() {
        let head = density_passthrough();
        let p = head.predict_memory(&query_with(100, 400));
        let expected = 400.0 / (8.0 * 100.0); // 0.5
        assert!((p - expected).abs() < 1e-15, "got {p}, want {expected}");
    }

    #[test]
    fn prediction_is_clamped_to_unit_interval() {
        let mut head = density_passthrough();
        head.intercept = 5.0;
        assert_eq!(head.predict_memory(&query_with(10, 10)), 1.0);
        head.intercept = -5.0;
        assert_eq!(head.predict_memory(&query_with(10, 10)), 0.0);
    }

    #[test]
    fn validity_rejects_nonfinite_and_zero_stds() {
        let mut head = density_passthrough();
        assert!(head.is_valid());
        head.feature_stds[6] = 0.0;
        assert!(!head.is_valid());
        let mut head2 = density_passthrough();
        head2.coefficients[0] = f64::NAN;
        assert!(!head2.is_valid());
        let mut head3 = density_passthrough();
        head3.intercept = f64::NEG_INFINITY;
        assert!(!head3.is_valid());
    }

    #[test]
    fn prediction_is_bit_deterministic() {
        let head = PinnMemoryV1 {
            feature_means: [3.1, 8.2, 0.5, 0.1, 0.2, 2.7, 4.4, 0.05],
            feature_stds: [1.5, 2.0, 0.7, 0.3, 0.4, 1.9, 2.2, 0.06],
            coefficients: [0.01, -0.02, 0.003, 0.0, 0.004, 0.12, 0.4, 0.31],
            intercept: 0.15,
        };
        let q = query_with(123_456, 78_900);
        let first = head.predict_memory(&q).to_bits();
        for _ in 0..50 {
            assert_eq!(first, head.predict_memory(&q).to_bits());
        }
    }
}
