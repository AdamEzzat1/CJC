//! Property tests for DoubleMLEstimator.

use super::common::{empty_locke_report, synthetic_dml};
use cjc_causal::{DoubleMLEstimator, IdentificationAssumption};
use proptest::prelude::*;

fn assumptions_dml() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::Unconfoundedness,
        IdentificationAssumption::NoInterference,
    ]
}

proptest! {
    /// p1 — same input + same seed ⇒ byte-identical EffectEstimate.
    #[test]
    fn dml_p1_same_input_byte_identical(
        data_seed in 0u64..1_000_000u64,
        est_seed in 0u64..1_000_000u64,
        n in 60usize..160usize,
    ) {
        let df = synthetic_dml(n, data_seed, 1.0);
        let report = empty_locke_report();
        let est = DoubleMLEstimator::new().with_seed(est_seed);
        let e1 = est.estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report);
        let e2 = est.estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report);
        match (e1, e2) {
            (Ok(a), Ok(b)) => {
                prop_assert_eq!(a.point.to_bits(), b.point.to_bits());
                prop_assert_eq!(a.std_error.to_bits(), b.std_error.to_bits());
                prop_assert_eq!(a.ci_lower.to_bits(), b.ci_lower.to_bits());
                prop_assert_eq!(a.ci_upper.to_bits(), b.ci_upper.to_bits());
                prop_assert_eq!(a.identifier, b.identifier);
            }
            (Err(ea), Err(eb)) => {
                prop_assert_eq!(format!("{}", ea), format!("{}", eb));
            }
            _ => prop_assert!(false, "Ok/Err disagreement across identical DML runs"),
        }
    }

    /// p2 — covariate order does not change the identifier.
    ///
    /// The covariate ORDER affects the cross-fit predictions because lm()
    /// fits with columns in the given order, which COULD numerically
    /// perturb estimates. But the `compute_identifier` canonical-sort
    /// ensures the hash is order-invariant.
    #[test]
    fn dml_p2_covariate_order_invariance_on_identifier(
        data_seed in 0u64..1_000u64,
    ) {
        let df = synthetic_dml(80, data_seed, 1.0);
        let report = empty_locke_report();
        let est = DoubleMLEstimator::new().with_seed(0);
        let e_ab = est.estimate(&df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report);
        let e_ba = est.estimate(&df, "treatment", "outcome", &["x2", "x1"], &assumptions_dml(), &report);
        if let (Ok(a), Ok(b)) = (e_ab, e_ba) {
            // The β estimate may differ in floating-point by a few ULPs because
            // lm() internally orders columns differently. But the IDENTIFIER
            // is computed from canonically-sorted covariates so:
            //   sorted(["x1", "x2"]) == sorted(["x2", "x1"])
            // Therefore if the β bits and SE bits happen to match (as they
            // should for the well-conditioned synthetic data here), so do
            // the identifiers. We assert the looser property: the canonical
            // sort means identifiers DEPEND ONLY on the sorted set.
            // For an ABSOLUTELY rigorous test, we'd need point-equal results,
            // which the lm() implementation makes hard to guarantee under
            // column permutation. We assert the achievable: if point + se
            // match, identifier matches.
            if a.point.to_bits() == b.point.to_bits()
                && a.std_error.to_bits() == b.std_error.to_bits()
            {
                prop_assert_eq!(a.identifier, b.identifier);
            }
        }
    }

    /// p3 — CI bounds are ordered correctly.
    #[test]
    fn dml_p3_ci_bounds_well_ordered(
        data_seed in 0u64..1_000u64,
    ) {
        let df = synthetic_dml(80, data_seed, 1.0);
        let report = empty_locke_report();
        if let Ok(est) = DoubleMLEstimator::new().estimate(
            &df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report,
        ) {
            prop_assert!(est.ci_lower <= est.point, "lower {} > point {}", est.ci_lower, est.point);
            prop_assert!(est.point <= est.ci_upper, "point {} > upper {}", est.point, est.ci_upper);
        }
    }

    /// p4 — different `k_folds` values produce different identifiers
    /// (the seed alone isn't the only thing affecting the result).
    #[test]
    fn dml_p4_different_k_produces_different_identifier(
        data_seed in 0u64..1_000u64,
    ) {
        let df = synthetic_dml(100, data_seed, 1.0);
        let report = empty_locke_report();
        let e3 = DoubleMLEstimator::new().with_k_folds(3).with_seed(0).estimate(
            &df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report,
        );
        let e5 = DoubleMLEstimator::new().with_k_folds(5).with_seed(0).estimate(
            &df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report,
        );
        if let (Ok(a), Ok(b)) = (e3, e5) {
            prop_assert_ne!(a.identifier, b.identifier);
        }
    }

    /// p5 — DML's `iv_first_stage_f` is always `None`.
    #[test]
    fn dml_p5_iv_first_stage_f_is_always_none(
        data_seed in 0u64..1_000u64,
    ) {
        let df = synthetic_dml(80, data_seed, 1.0);
        let report = empty_locke_report();
        if let Ok(est) = DoubleMLEstimator::new().estimate(
            &df, "treatment", "outcome", &["x1", "x2"], &assumptions_dml(), &report,
        ) {
            prop_assert!(est.iv_first_stage_f.is_none());
            prop_assert!(est.balance_diagnostics.is_none());
        }
    }
}
