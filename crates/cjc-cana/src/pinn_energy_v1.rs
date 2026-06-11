//! PINN energy head v1 — the TRAINED plan-energy predictor (Phase B).
//!
//! ## What this is
//!
//! A deterministic linear (ridge) model predicting `ln(score)` — the
//! log of the baseline-relative modeled energy of running a program
//! under a candidate optimization plan. Fit OFFLINE by
//! `bench/cana_train_pinn -- train-energy` over the schema-v3 ablation
//! corpus; persisted as a CPB1 bundle
//! (`cjc-cana-compress::energy_bundle`); loaded read-only (training
//! never runs during compilation).
//!
//! ## Evidence-chosen recipe (sanity-energy pass, 2026-06-11)
//!
//! Every design choice below was selected by measurement, not theory
//! (full grid in `docs/cana/PINN_V2_DESIGN.md` §9):
//!
//! - **Target `ln(score)`** — raw score's heavy right tail (max 11.16)
//!   made every raw-target fit fail held-out (R²(test) −1.3 to −18).
//! - **No config one-hots** — the −32 R²(test) failure replicated on
//!   the v3 corpus (−16.98) with one-hots present; they are collinear
//!   with pass counts AND unavailable to the deployed consumer (a
//!   plan selector scores candidate plans, which have no config id).
//! - **Loop + structural features both required** — diverged-rows
//!   R²(test): base −0.36, +loops 0.45, +structural 0.34, both 0.82.
//! - **Fit on ALL rows (ties included)** — the diverged-only fit has
//!   the better R² but WORSE selector regret than the always-baseline
//!   heuristic (+0.051 vs +0.033); the all-rows fit reaches +0.0014
//!   test regret (32/34 exact-best picks, 10/10 frozen holdout).
//!
//! ## Feature basis (single definition — trainer and selector share it)
//!
//! | idx | feature |
//! |-----|---------|
//! | 0 | `ln(1 + flops_estimate)` |
//! | 1 | `ln(1 + bytes_read_estimate)` |
//! | 2 | `ln(1 + bytes_written_estimate)` |
//! | 3 | `ln(1 + allocation_bytes_estimate)` |
//! | 4 | `ln(1 + working_set_bytes_estimate)` |
//! | 5 | `ln(1 + float_ops_estimate)` |
//! | 6 | FP density `min(float_ops/flops, 1)` (0 when flops = 0) |
//! | 7 | `ln(1 + mir_nodes_before)` |
//! | 8 | `recommended_count` |
//! | 9 | `dropped_count` |
//! | 10..10+P | per-pass plan counts, aligned with [`PinnEnergyV1::pass_names`] |
//! | 10+P | `ln(1 + countable_loop_count_total)` |
//! | 11+P | `max_loop_depth` (max across functions) |
//! | 12+P | `ln(1 + mir_nodes_after)` (statically computable: apply the plan) |
//! | 13+P | `mir_nodes_after / mir_nodes_before` (1.0 when before = 0) |
//!
//! The pass-name list is part of the trained artifact (stored in the
//! CPB1 bundle), so a corpus whose plan vocabulary drifts produces a
//! NEW head rather than silently misaligned counts.
//!
//! ## Determinism
//!
//! `predict_ln_score` is a pure fixed-order dot product with named
//! intermediates (no FMA). Model identity (`model_id`,
//! `model_version`) flows into report hashes when a consumer activates
//! the head (Phase C); training is offline-only; shadow gating
//! precedes any activation (invariants unchanged).

use crate::hash::CanaHasher;

/// Stable model identifier for report hashes.
pub const PINN_ENERGY_V1_MODEL_ID: &str = "pinn_energy_v1";

/// Monotonic version; bump on any change to the basis, the
/// standardization scheme, or the training recipe.
pub const PINN_ENERGY_V1_MODEL_VERSION: u32 = 1;

/// Number of features BEFORE the pass-count block.
pub const ENERGY_WORKLOAD_FEATURES: usize = 10;

/// Number of features AFTER the pass-count block.
pub const ENERGY_TAIL_FEATURES: usize = 4;

/// Plain-data inputs to the energy basis — both the trainer (lifting
/// corpus rows) and the future plan selector (lifting candidate plans)
/// construct this, so the basis definition cannot drift between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyQuery {
    pub flops_estimate: u64,
    pub bytes_read_estimate: u64,
    pub bytes_written_estimate: u64,
    pub allocation_bytes_estimate: u64,
    pub working_set_bytes_estimate: u64,
    pub float_ops_estimate: u64,
    pub mir_nodes_before: u64,
    pub recommended_count: u32,
    pub dropped_count: u32,
    /// Occurrences of each pass in the candidate plan, aligned with
    /// the HEAD's `pass_names` (build via [`PinnEnergyV1::pass_counts`]).
    pub pass_counts: Vec<u32>,
    /// Total statically-countable loops across functions.
    pub countable_loop_count: u64,
    /// Max loop nesting depth across functions.
    pub max_loop_depth: u32,
    /// Post-plan node count (statically computable by applying the
    /// plan — cheap and deterministic).
    pub mir_nodes_after: u64,
}

/// Trained linear energy head: pass vocabulary + standardization +
/// ridge coefficients over the documented basis.
#[derive(Debug, Clone, PartialEq)]
pub struct PinnEnergyV1 {
    /// Pass names defining the count block, in basis order. Part of
    /// the trained artifact (persisted in CPB1).
    pub pass_names: Vec<String>,
    /// Per-feature training means (z-score centering).
    pub feature_means: Vec<f64>,
    /// Per-feature training stds (zero-variance stored as 1.0).
    pub feature_stds: Vec<f64>,
    /// Ridge coefficients over standardized features.
    pub coefficients: Vec<f64>,
    /// Intercept.
    pub intercept: f64,
}

impl PinnEnergyV1 {
    /// Total feature count for this head's basis.
    pub fn feature_count(&self) -> usize {
        ENERGY_WORKLOAD_FEATURES + self.pass_names.len() + ENERGY_TAIL_FEATURES
    }

    /// `true` iff dimensions agree, every parameter is finite, and
    /// every std is strictly positive. Invalid heads must never be
    /// consulted — loaders reject at the boundary.
    pub fn is_valid(&self) -> bool {
        let n = self.feature_count();
        let dims_ok = self.feature_means.len() == n
            && self.feature_stds.len() == n
            && self.coefficients.len() == n;
        let finite = self
            .feature_means
            .iter()
            .chain(self.feature_stds.iter())
            .chain(self.coefficients.iter())
            .all(|v| v.is_finite())
            && self.intercept.is_finite();
        let stds_positive = self.feature_stds.iter().all(|s| *s > 0.0);
        dims_ok && finite && stds_positive
    }

    /// Map a plan's pass list onto this head's count block.
    pub fn pass_counts<'a, I: IntoIterator<Item = &'a str>>(&self, passes: I) -> Vec<u32> {
        let mut counts = vec![0u32; self.pass_names.len()];
        for p in passes {
            if let Some(i) = self.pass_names.iter().position(|n| n == p) {
                counts[i] = counts[i].saturating_add(1);
            }
            // Unknown passes contribute nothing — under-counting is
            // the conservative direction (plan looks more neutral).
        }
        counts
    }

    /// Build the feature vector for one query. THE basis definition —
    /// see the module-docs table.
    pub fn features_from_query(&self, q: &EnergyQuery) -> Vec<f64> {
        let density = if q.flops_estimate == 0 {
            0.0
        } else {
            (q.float_ops_estimate as f64 / q.flops_estimate as f64).min(1.0)
        };
        let mut x = Vec::with_capacity(self.feature_count());
        x.push(log1p_u64(q.flops_estimate));
        x.push(log1p_u64(q.bytes_read_estimate));
        x.push(log1p_u64(q.bytes_written_estimate));
        x.push(log1p_u64(q.allocation_bytes_estimate));
        x.push(log1p_u64(q.working_set_bytes_estimate));
        x.push(log1p_u64(q.float_ops_estimate));
        x.push(density);
        x.push(log1p_u64(q.mir_nodes_before));
        x.push(q.recommended_count as f64);
        x.push(q.dropped_count as f64);
        for i in 0..self.pass_names.len() {
            x.push(q.pass_counts.get(i).copied().unwrap_or(0) as f64);
        }
        x.push(log1p_u64(q.countable_loop_count));
        x.push(q.max_loop_depth as f64);
        x.push(log1p_u64(q.mir_nodes_after));
        let ratio = if q.mir_nodes_before > 0 {
            q.mir_nodes_after as f64 / q.mir_nodes_before as f64
        } else {
            1.0
        };
        x.push(ratio);
        x
    }

    /// Predict `ln(score)` for one query. Lower = the plan costs less
    /// modeled energy than the baseline plan; 0 = tie. Fixed-order
    /// standardize → dot (named intermediates, no FMA). NaN-safe: a
    /// non-finite accumulation returns 0 (= "predict tie"), never NaN.
    pub fn predict_ln_score(&self, q: &EnergyQuery) -> f64 {
        let x = self.features_from_query(q);
        let n = self.feature_count();
        // Dimension guard covers the PARAMETER vectors too — a head
        // whose arrays drifted from its vocabulary must predict the
        // neutral 0 ("tie"), never index out of bounds.
        if x.len() != n
            || self.feature_means.len() != n
            || self.feature_stds.len() != n
            || self.coefficients.len() != n
        {
            return 0.0;
        }
        let mut acc = self.intercept;
        for i in 0..n {
            let centered = x[i] - self.feature_means[i];
            let scaled = centered / self.feature_stds[i];
            let term = self.coefficients[i] * scaled;
            acc += term;
        }
        if acc.is_finite() {
            acc
        } else {
            0.0
        }
    }

    /// Feed identity + parameters into a streaming hasher (report-hash
    /// integration for Phase C consumers).
    pub fn feed(&self, hasher: &mut CanaHasher) {
        hasher.write_tag(TAG_ENERGY_HEAD);
        hasher.write_u32(PINN_ENERGY_V1_MODEL_VERSION);
        hasher.write_u32(self.pass_names.len() as u32);
        for name in &self.pass_names {
            hasher.write_str(name);
        }
        for v in self
            .feature_means
            .iter()
            .chain(self.feature_stds.iter())
            .chain(self.coefficients.iter())
        {
            hasher.write_u64(v.to_bits());
        }
        hasher.write_u64(self.intercept.to_bits());
    }
}

/// Discriminator tag (tag-space convention: 0xA0 memory, 0xB0
/// reductions, 0xC0 type-mix, 0xD0 energy head).
const TAG_ENERGY_HEAD: u8 = 0xD0;

fn log1p_u64(v: u64) -> f64 {
    (v as f64).ln_1p()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn passthrough_head(pass_names: &[&str]) -> PinnEnergyV1 {
        let n = ENERGY_WORKLOAD_FEATURES + pass_names.len() + ENERGY_TAIL_FEATURES;
        PinnEnergyV1 {
            pass_names: pass_names.iter().map(|s| s.to_string()).collect(),
            feature_means: vec![0.0; n],
            feature_stds: vec![1.0; n],
            coefficients: vec![0.0; n],
            intercept: 0.0,
        }
    }

    fn query(head: &PinnEnergyV1, passes: &[&str]) -> EnergyQuery {
        EnergyQuery {
            flops_estimate: 1000,
            bytes_read_estimate: 8000,
            bytes_written_estimate: 2000,
            allocation_bytes_estimate: 64,
            working_set_bytes_estimate: 1024,
            float_ops_estimate: 250,
            mir_nodes_before: 120,
            recommended_count: 3,
            dropped_count: 1,
            pass_counts: head.pass_counts(passes.iter().copied()),
            countable_loop_count: 2,
            max_loop_depth: 1,
            mir_nodes_after: 95,
        }
    }

    #[test]
    fn feature_count_tracks_pass_vocabulary() {
        let head = passthrough_head(&["dce", "licm", "loop_unroll"]);
        assert_eq!(head.feature_count(), 10 + 3 + 4);
        assert!(head.is_valid());
    }

    #[test]
    fn pass_counts_align_and_ignore_unknown() {
        let head = passthrough_head(&["dce", "licm"]);
        let counts = head.pass_counts(["licm", "dce", "licm", "mystery_pass"]);
        assert_eq!(counts, vec![1, 2]);
    }

    #[test]
    fn zero_coefficients_predict_intercept() {
        let mut head = passthrough_head(&["dce"]);
        head.intercept = 0.25;
        let q = query(&head, &["dce"]);
        assert_eq!(head.predict_ln_score(&q), 0.25);
    }

    #[test]
    fn dimension_mismatch_is_invalid_not_panic() {
        let mut head = passthrough_head(&["dce"]);
        head.coefficients.pop();
        assert!(!head.is_valid());
        // predict on an invalid head still must not panic.
        let q = query(&passthrough_head(&["dce"]), &["dce"]);
        let _ = head.predict_ln_score(&q);
    }

    #[test]
    fn validity_rejects_nonfinite_and_zero_stds() {
        let mut head = passthrough_head(&["dce"]);
        head.feature_stds[3] = 0.0;
        assert!(!head.is_valid());
        let mut head2 = passthrough_head(&["dce"]);
        head2.coefficients[0] = f64::NAN;
        assert!(!head2.is_valid());
    }

    #[test]
    fn density_caps_and_zero_flops_is_zero() {
        let head = passthrough_head(&[]);
        let mut q = query(&head, &[]);
        q.flops_estimate = 0;
        let x = head.features_from_query(&q);
        assert_eq!(x[6], 0.0);
        q.flops_estimate = 10;
        q.float_ops_estimate = 1000;
        let x2 = head.features_from_query(&q);
        assert_eq!(x2[6], 1.0);
    }

    #[test]
    fn prediction_is_bit_deterministic() {
        let mut head = passthrough_head(&["dce", "licm"]);
        head.coefficients = (0..head.feature_count())
            .map(|i| 0.01 * (i as f64 + 1.0))
            .collect();
        head.intercept = -0.05;
        let q = query(&head, &["dce", "licm", "dce"]);
        let first = head.predict_ln_score(&q).to_bits();
        for _ in 0..50 {
            assert_eq!(first, head.predict_ln_score(&q).to_bits());
        }
    }
}
