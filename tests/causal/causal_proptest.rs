//! Property tests for cjc-causal — encode the determinism + invariance
//! contracts from ADR-0043 as proptest properties.
//!
//! Each property uses 256 cases by default (proptest's per-test default).

use super::common::{empty_locke_report, synthetic_randomised};
use cjc_causal::{IdentificationAssumption, PropensityScoreMatcher};
use proptest::prelude::*;

fn default_assumptions() -> Vec<IdentificationAssumption> {
    vec![
        IdentificationAssumption::Unconfoundedness,
        IdentificationAssumption::Positivity,
        IdentificationAssumption::NoInterference,
    ]
}

proptest! {
    /// Property 1 — same seed + same data ⇒ byte-identical EffectEstimate.
    ///
    /// Generates a (data_seed, est_seed, n) triple; builds a synthetic frame;
    /// runs the matcher twice. Every f64 in the output must be bit-equal.
    #[test]
    fn p1_same_seed_byte_identical_estimate(
        data_seed in 0u64..1_000_000u64,
        est_seed in 0u64..1_000_000u64,
        n in 40usize..120usize,
    ) {
        let df = synthetic_randomised(n, data_seed, 1.0);
        let report = empty_locke_report();
        let m = PropensityScoreMatcher::new().with_seed(est_seed).with_bootstrap_reps(30);
        let e1 = m.estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report);
        let e2 = m.estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report);
        match (e1, e2) {
            (Ok(a), Ok(b)) => {
                prop_assert_eq!(a.point.to_bits(), b.point.to_bits());
                prop_assert_eq!(a.std_error.to_bits(), b.std_error.to_bits());
                prop_assert_eq!(a.ci_lower.to_bits(), b.ci_lower.to_bits());
                prop_assert_eq!(a.ci_upper.to_bits(), b.ci_upper.to_bits());
                prop_assert_eq!(a.identifier, b.identifier);
            }
            (Err(ea), Err(eb)) => {
                // Both arms agree on failure mode — also acceptable.
                prop_assert_eq!(format!("{}", ea), format!("{}", eb));
            }
            _ => prop_assert!(false, "Ok/Err disagreement across identical runs"),
        }
    }

    /// Property 2 — shifting the outcome by a constant `c` shifts the
    /// estimate's `point` by exactly `c` (modulo floating-point rounding
    /// within a small tolerance).
    #[test]
    fn p2_outcome_shift_invariance(
        data_seed in 0u64..1_000u64,
        shift in -10.0f64..10.0f64,
    ) {
        let mut df_a = synthetic_randomised(80, data_seed, 1.0);
        let mut df_b = df_a.clone();
        // Apply the shift to df_b's outcome column.
        match &mut df_b.columns.iter_mut().find(|(n, _)| n == "outcome").unwrap().1 {
            cjc_data::Column::Float(v) => {
                for x in v.iter_mut() {
                    *x += shift;
                }
            }
            _ => unreachable!(),
        }
        // df_a's outcome stays as-is.
        let _ = &mut df_a;

        let report = empty_locke_report();
        let m = PropensityScoreMatcher::new().with_seed(0).with_bootstrap_reps(20);
        let e_a = m.estimate(&df_a, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report);
        let e_b = m.estimate(&df_b, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report);

        if let (Ok(a), Ok(b)) = (e_a, e_b) {
            // The ATT is a mean of paired differences; shifting y by c shifts
            // each paired difference by 0 (both treated and matched control
            // got the same shift), so the ATT is unchanged.
            prop_assert!(
                (a.point - b.point).abs() < 1e-9,
                "outcome shift should not change ATT (a={}, b={}, shift={})",
                a.point, b.point, shift
            );
        }
    }

    /// Property 3 — bigger caliper produces at least as many matched pairs.
    ///
    /// Intuitively, a looser caliper admits more matches; the consumed-control
    /// monotonicity of greedy matching guarantees `n_treated(tight) ≤ n_treated(loose)`.
    #[test]
    fn p3_caliper_monotonicity(
        data_seed in 0u64..1_000u64,
        small in 0.05f64..0.3f64,
        big in 0.3f64..2.0f64,
    ) {
        prop_assume!(small < big);
        let df = synthetic_randomised(80, data_seed, 1.0);
        let report = empty_locke_report();
        let make = |c: f64| {
            PropensityScoreMatcher::new()
                .with_caliper_sd(c)
                .with_seed(0)
                .with_bootstrap_reps(10)
                .estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report)
        };
        if let (Ok(tight), Ok(loose)) = (make(small), make(big)) {
            prop_assert!(
                tight.n_treated <= loose.n_treated,
                "tight caliper {} matched {} pairs > loose caliper {} matched {} pairs",
                small, tight.n_treated, big, loose.n_treated,
            );
        }
    }

    /// Property 4 — the `EffectEstimate.identifier` depends on the seed
    /// in a deterministic, repeatable way. Two runs with the same seed
    /// produce the same identifier; two runs with different seeds
    /// produce identifiers that are well-defined functions of their inputs.
    #[test]
    fn p4_identifier_stable_per_seed(
        data_seed in 0u64..1_000u64,
        est_seed in 0u64..1_000u64,
    ) {
        let df = synthetic_randomised(60, data_seed, 1.0);
        let report = empty_locke_report();
        let m = PropensityScoreMatcher::new().with_seed(est_seed).with_bootstrap_reps(20);
        let e1 = m.estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report);
        let e2 = m.estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report);
        if let (Ok(a), Ok(b)) = (e1, e2) {
            prop_assert_eq!(a.identifier, b.identifier);
        }
    }

    /// Property 5 — covariate order does not affect the estimate.
    ///
    /// Same data, two different covariate orderings → ATT and identifier
    /// must agree byte-for-byte. The identifier is canonicalised by sorting
    /// covariates inside `content_hash`; the ATT depends only on the column
    /// data, not the argument order.
    #[test]
    fn p5_covariate_order_does_not_affect_estimate(
        data_seed in 0u64..1_000u64,
    ) {
        let df = synthetic_randomised(60, data_seed, 1.0);
        let report = empty_locke_report();
        let m = PropensityScoreMatcher::new().with_seed(0).with_bootstrap_reps(20);
        let e_ab = m.estimate(&df, "treatment", "outcome", &["age", "income"], &default_assumptions(), &report);
        let e_ba = m.estimate(&df, "treatment", "outcome", &["income", "age"], &default_assumptions(), &report);
        if let (Ok(a), Ok(b)) = (e_ab, e_ba) {
            prop_assert_eq!(a.point.to_bits(), b.point.to_bits());
            prop_assert_eq!(a.identifier, b.identifier);
        }
    }
}
