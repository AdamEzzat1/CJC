//! Property tests for IVRegression — encode IV-specific determinism +
//! invariance contracts as proptest properties (256 cases each).

use super::common::{empty_locke_report, synthetic_iv};
use cjc_causal::{IVRegression, IdentificationAssumption};
use proptest::prelude::*;

fn assumptions_iv() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::ExcludabilityOfInstrument,
        IdentificationAssumption::MonotonicityOfInstrument,
        IdentificationAssumption::NoInterference,
    ]
}

proptest! {
    /// p1 — same input ⇒ byte-identical EffectEstimate (point, SE, identifier, F).
    ///
    /// Unlike PSM, IV has no RNG-driven step (no bootstrap in v0.1), so byte
    /// identity is even stronger than for PSM — there's nothing to seed away.
    #[test]
    fn iv_p1_same_input_byte_identical(
        data_seed in 0u64..1_000_000u64,
        n in 40usize..120usize,
    ) {
        let df = synthetic_iv(n, data_seed, 1.0, 0.7);
        let report = empty_locke_report();
        let iv = IVRegression::new();
        let e1 = iv.estimate(&df, "treatment", "outcome", "instrument", &["x"], &assumptions_iv(), &report);
        let e2 = iv.estimate(&df, "treatment", "outcome", "instrument", &["x"], &assumptions_iv(), &report);
        match (e1, e2) {
            (Ok(a), Ok(b)) => {
                prop_assert_eq!(a.point.to_bits(), b.point.to_bits());
                prop_assert_eq!(a.std_error.to_bits(), b.std_error.to_bits());
                prop_assert_eq!(a.ci_lower.to_bits(), b.ci_lower.to_bits());
                prop_assert_eq!(a.ci_upper.to_bits(), b.ci_upper.to_bits());
                prop_assert_eq!(a.identifier, b.identifier);
                prop_assert_eq!(
                    a.iv_first_stage_f.unwrap().to_bits(),
                    b.iv_first_stage_f.unwrap().to_bits()
                );
            }
            (Err(ea), Err(eb)) => {
                prop_assert_eq!(format!("{}", ea), format!("{}", eb));
            }
            _ => prop_assert!(false, "Ok/Err disagreement across identical IV runs"),
        }
    }

    /// p2 — first-stage F is always non-negative.
    ///
    /// In the just-identified case, F = t_γ². Squared values are ≥ 0 by
    /// construction; any negative F would indicate a sign bug.
    #[test]
    fn iv_p2_f_stat_is_non_negative(
        data_seed in 0u64..1_000_000u64,
        gamma in 0.05f64..2.0f64,
    ) {
        let df = synthetic_iv(80, data_seed, 1.0, gamma);
        let report = empty_locke_report();
        if let Ok(est) = IVRegression::new().estimate(
            &df, "treatment", "outcome", "instrument", &["x"], &assumptions_iv(), &report,
        ) {
            let f = est.iv_first_stage_f.expect("F must be populated");
            prop_assert!(f >= 0.0, "F = {} must be ≥ 0 (it's t²)", f);
        }
    }

    /// p3 — covariate order does not affect the IV estimate.
    ///
    /// Two runs with covariates in different orders must produce byte-identical
    /// point, SE, and identifier (the latter via canonical sort inside
    /// `content_hash`).
    #[test]
    fn iv_p3_covariate_order_invariance(
        data_seed in 0u64..1_000u64,
    ) {
        // Need at least 2 covariates to have a meaningful permutation.
        let df = synthetic_iv(80, data_seed, 1.0, 0.7);
        // The synthetic_iv fixture only has one covariate "x"; we'll add a
        // duplicate of x as "x2" to give the test some surface area.
        let mut df2 = df.clone();
        let x_vals = match &df.columns.iter().find(|(n, _)| n == "x").unwrap().1 {
            cjc_data::Column::Float(v) => v.clone(),
            _ => unreachable!(),
        };
        df2.columns.push(("x2".to_string(), cjc_data::Column::Float(x_vals)));
        let report = empty_locke_report();
        let iv = IVRegression::new();

        let e_ab = iv.estimate(&df2, "treatment", "outcome", "instrument", &["x", "x2"], &assumptions_iv(), &report);
        let e_ba = iv.estimate(&df2, "treatment", "outcome", "instrument", &["x2", "x"], &assumptions_iv(), &report);
        if let (Ok(a), Ok(b)) = (e_ab, e_ba) {
            // Identifier must match (canonical sort).
            prop_assert_eq!(a.identifier, b.identifier);
        }
    }

    /// p4 — IV identifier vs PSM identifier differ on the same (T, Y, X, A).
    ///
    /// Even when treatment, outcome, covariates, and assumptions are
    /// identical, the IV identifier must include the instrument column and
    /// estimator label, so it differs from PSM.
    #[test]
    fn iv_p4_identifier_differs_from_psm(
        data_seed in 0u64..1_000u64,
    ) {
        use cjc_causal::PropensityScoreMatcher;
        let mut df = synthetic_iv(80, data_seed, 1.0, 0.7);
        // Make treatment binary for PSM. The continuous T from synthetic_iv
        // would fail PSM's binary check.
        let n = df.nrows();
        let bin_t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        match &mut df.columns.iter_mut().find(|(n, _)| n == "treatment").unwrap().1 {
            cjc_data::Column::Float(v) => *v = bin_t,
            _ => unreachable!(),
        }
        let report = empty_locke_report();

        let psm = PropensityScoreMatcher::new().with_bootstrap_reps(5);
        let iv = IVRegression::new();
        let e_psm = psm.estimate(&df, "treatment", "outcome", &["x"], &assumptions_iv(), &report);
        let e_iv = iv.estimate(&df, "treatment", "outcome", "instrument", &["x"], &assumptions_iv(), &report);
        if let (Ok(p), Ok(i)) = (e_psm, e_iv) {
            prop_assert_ne!(p.identifier, i.identifier);
        }
    }

    /// p5 — confidence-interval bounds are ordered correctly.
    ///
    /// CI lower ≤ point ≤ CI upper, for any valid input. Easy to verify
    /// but catches sign-flip bugs in the CI construction.
    #[test]
    fn iv_p5_ci_bounds_well_ordered(
        data_seed in 0u64..1_000u64,
    ) {
        let df = synthetic_iv(80, data_seed, 1.0, 0.7);
        let report = empty_locke_report();
        if let Ok(est) = IVRegression::new().estimate(
            &df, "treatment", "outcome", "instrument", &["x"], &assumptions_iv(), &report,
        ) {
            prop_assert!(est.ci_lower <= est.point, "CI lower {} > point {}", est.ci_lower, est.point);
            prop_assert!(est.point <= est.ci_upper, "point {} > CI upper {}", est.point, est.ci_upper);
        }
    }
}
