//! PINN v2 — the TRAINED thermal head.
//!
//! ## What v2 is (and deliberately is not)
//!
//! The §2.1 data-sanity pass over the 1,474-row ablation corpus
//! (see `docs/cana/PINN_V2_DESIGN.md`) found:
//!
//! - Recorded thermal became **linearly** predictable the moment the
//!   type-blind feature gap was closed (`TypeMix` → `float_ops_estimate`):
//!   held-out R² jumped from −0.05 to ≈0.96. Linear saturates; with
//!   n=134 programs an MLP chasing the residual would overfit.
//! - cpu / memory / energy labels lack the range or variance to train
//!   on (cpu std 0.013, memory std 0.0007, 39 informative energy rows).
//!
//! So v2 = **the v1 closed form + this trained linear thermal head**.
//! The weights are fit OFFLINE by `bench/cana_train_pinn` (deterministic
//! closed-form ridge regression — no RNG, no SGD, bit-reproducible) and
//! persisted as a CPB0 bundle (`cjc-cana-compress::pinn_bundle`).
//! Compilation only ever LOADS weights (determinism invariant: training
//! offline only).
//!
//! ## Feature basis
//!
//! Seven features, all derivable from a [`PhysicalCostQuery`] (so
//! train-time rows and predict-time queries share one basis):
//!
//! | idx | feature |
//! |-----|---------|
//! | 0 | `ln(1 + flops_estimate)` |
//! | 1 | `ln(1 + bytes_read_estimate)` |
//! | 2 | `ln(1 + bytes_written_estimate)` |
//! | 3 | `ln(1 + allocation_bytes_estimate)` |
//! | 4 | `ln(1 + working_set_bytes_estimate)` |
//! | 5 | `ln(1 + float_ops_estimate)` |
//! | 6 | `float_ops_estimate / flops_estimate` (FP density — the static analog of recorded `thermal_intensity`) |
//!
//! Granularity note: training rows hold per-PROGRAM sums (neutral-pass
//! queries) while prediction runs per-FUNCTION with pass amplification.
//! The density feature (the dominant signal, corr ≈ +0.95 with the
//! label) is amplification-invariant — numerator and denominator carry
//! the same `loop_amp × pass_amp` factor — and the log magnitudes shift
//! by only `ln(amp)`. The shadow-mode gate measures the consequence of
//! this approximation against real labels before promotion.
//!
//! ## Determinism
//!
//! `predict_thermal` is a pure function of `(weights, query)`. The dot
//! product is fixed-order with named intermediates (no FMA, invariant
//! #3). Weights carry `model_id`/`model_version` into report hashes via
//! `CostModel::name()/version()` exactly like v1.

use crate::physical_cost::PhysicalCostQuery;

/// Stable v2 model identifier; flows into report hashes via
/// [`crate::cost_model::CostModel::name`] when a trained head is
/// attached.
pub const PINN_V2_MODEL_ID: &str = "pinn_thermal_v2";

/// Monotonic v2 model version; bump on any change to the feature
/// basis, the standardization scheme, or the training recipe.
pub const PINN_V2_MODEL_VERSION: u32 = 2;

/// Number of features in the v2 basis (see module docs table).
pub const PINN_V2_FEATURE_COUNT: usize = 7;

/// Trained linear thermal head: standardization parameters + ridge
/// coefficients, fit offline by `bench/cana_train_pinn`.
#[derive(Debug, Clone, PartialEq)]
pub struct PinnThermalV2 {
    /// Per-feature training means (z-score centering).
    pub feature_means: [f64; PINN_V2_FEATURE_COUNT],
    /// Per-feature training standard deviations (z-score scaling).
    /// Zero-variance features are stored as 1.0 by the trainer.
    pub feature_stds: [f64; PINN_V2_FEATURE_COUNT],
    /// Ridge coefficients over the standardized features.
    pub coefficients: [f64; PINN_V2_FEATURE_COUNT],
    /// Intercept term.
    pub intercept: f64,
}

impl PinnThermalV2 {
    /// `true` iff every parameter is finite and every std is strictly
    /// positive. Invalid heads must never be attached to a cost model —
    /// loaders reject them at the boundary.
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

    /// Predict thermal pressure in `[0, 1]` for one workload query.
    ///
    /// Fixed-order standardize → dot → clamp; no FMA (each product is
    /// bound to a named intermediate before any addition).
    pub fn predict_thermal(&self, query: &PhysicalCostQuery<'_>) -> f64 {
        let x = features_from_query(query);
        let mut acc = self.intercept;
        for i in 0..PINN_V2_FEATURE_COUNT {
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

/// Build the v2 feature vector from a workload query. The single
/// definition both the trainer (over row-level sums lifted into a
/// synthetic query) and the predictor use — drift between the two
/// would silently invalidate the weights.
pub fn features_from_query(q: &PhysicalCostQuery<'_>) -> [f64; PINN_V2_FEATURE_COUNT] {
    let density = if q.flops_estimate == 0 {
        0.0
    } else {
        q.float_ops_estimate as f64 / q.flops_estimate as f64
    };
    [
        log1p_u64(q.flops_estimate),
        log1p_u64(q.bytes_read_estimate),
        log1p_u64(q.bytes_written_estimate),
        log1p_u64(q.allocation_bytes_estimate),
        log1p_u64(q.working_set_bytes_estimate),
        log1p_u64(q.float_ops_estimate),
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

    fn query_with(flops: u64, float_ops: u64) -> PhysicalCostQuery<'static> {
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
            float_ops_estimate: float_ops,
        }
    }

    /// A head that passes the FP density through unchanged: only
    /// feature 6 has weight, identity standardization.
    fn density_passthrough() -> PinnThermalV2 {
        PinnThermalV2 {
            feature_means: [0.0; PINN_V2_FEATURE_COUNT],
            feature_stds: [1.0; PINN_V2_FEATURE_COUNT],
            coefficients: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            intercept: 0.0,
        }
    }

    #[test]
    fn density_feature_is_amplification_invariant() {
        let a = features_from_query(&query_with(100, 25));
        let b = features_from_query(&query_with(1000, 250));
        assert_eq!(a[6], b[6], "density must not depend on amplification");
        assert!((a[6] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn zero_flops_density_is_zero_not_nan() {
        let x = features_from_query(&query_with(0, 0));
        assert_eq!(x[6], 0.0);
    }

    #[test]
    fn passthrough_head_predicts_density() {
        let head = density_passthrough();
        let p = head.predict_thermal(&query_with(1000, 700));
        assert!((p - 0.7).abs() < 1e-15, "got {p}");
    }

    #[test]
    fn prediction_is_clamped_to_unit_interval() {
        let mut head = density_passthrough();
        head.intercept = 5.0;
        assert_eq!(head.predict_thermal(&query_with(10, 10)), 1.0);
        head.intercept = -5.0;
        assert_eq!(head.predict_thermal(&query_with(10, 10)), 0.0);
    }

    #[test]
    fn validity_rejects_nonfinite_and_zero_stds() {
        let mut head = density_passthrough();
        assert!(head.is_valid());
        head.feature_stds[3] = 0.0;
        assert!(!head.is_valid());
        let mut head2 = density_passthrough();
        head2.coefficients[0] = f64::NAN;
        assert!(!head2.is_valid());
        let mut head3 = density_passthrough();
        head3.intercept = f64::INFINITY;
        assert!(!head3.is_valid());
    }

    #[test]
    fn prediction_is_bit_deterministic() {
        let head = PinnThermalV2 {
            feature_means: [3.1, 8.2, 0.5, 0.1, 0.2, 2.7, 0.05],
            feature_stds: [1.5, 2.0, 0.7, 0.3, 0.4, 1.9, 0.06],
            coefficients: [0.01, -0.02, 0.003, 0.0, 0.004, 0.12, 0.31],
            intercept: 0.15,
        };
        let q = query_with(123_456, 7_890);
        let first = head.predict_thermal(&q).to_bits();
        for _ in 0..50 {
            assert_eq!(first, head.predict_thermal(&q).to_bits());
        }
    }
}
