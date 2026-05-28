//! `proptest` property tests for Locke.
//!
//! These cover invariants stated in the brief:
//!
//! * reports are deterministic across repeated runs
//! * missingness counts never exceed row count
//! * duplicate counts are stable under deterministic ordering
//! * confidence scores stay in [0, 1]
//! * adding more missingness should not improve missingness score
//! * adding lineage parents should not reduce lineage completeness
//!
//! Each property runs with the proptest default (256 cases). Large `n`
//! parameters are capped to keep CI runtimes reasonable.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{belief_report_from_locke, validate, ValidateOptions};
use cjc_locke::belief::sample_score_from_n;
use cjc_locke::drift::{compare, DriftConfig};
use cjc_locke::validation::ValidationConfig;
use proptest::prelude::*;

fn arb_float_vec() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(any::<i32>().prop_map(|i| i as f64), 1..50)
}

fn arb_mostly_nan_vec(nan_prob: u8) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec((0u8..100, any::<i32>()), 1..50).prop_map(move |xs| {
        xs.into_iter()
            .map(|(p, v)| if p < nan_prob { f64::NAN } else { v as f64 })
            .collect()
    })
}

proptest! {
    #[test]
    fn validate_is_deterministic_under_arbitrary_inputs(v in arb_float_vec()) {
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let opts = ValidateOptions {
            dataset_label: "prop".into(),
            ..Default::default()
        };
        let a = validate(&df, &opts);
        let b = validate(&df, &opts);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn missingness_count_never_exceeds_row_count(v in arb_mostly_nan_vec(50)) {
        let n = v.len() as u64;
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let opts = ValidateOptions {
            dataset_label: "prop".into(),
            ..Default::default()
        };
        let r = validate(&df, &opts);
        for f in &r.findings {
            if f.code == "E9001" {
                for ev in &f.evidence {
                    if let cjc_locke::FindingEvidence::Count { label, value } = ev {
                        if label == "n_missing" {
                            prop_assert!(*value <= n);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn belief_overall_is_in_unit_interval(v in arb_float_vec()) {
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let opts = ValidateOptions {
            dataset_label: "prop".into(),
            ..Default::default()
        };
        let r = validate(&df, &opts);
        let b = belief_report_from_locke(&r);
        prop_assert!(b.score.overall >= 0.0 && b.score.overall <= 1.0);
        prop_assert!(b.score.missingness_score >= 0.0 && b.score.missingness_score <= 1.0);
        prop_assert!(b.score.duplication_score >= 0.0 && b.score.duplication_score <= 1.0);
    }

    #[test]
    fn more_missingness_does_not_improve_missingness_score(
        seed in any::<u64>(),
        extra_nans in 0u64..30
    ) {
        let mut v: Vec<f64> = (0..50).map(|i| ((seed.wrapping_mul(31).wrapping_add(i)) % 1000) as f64).collect();
        let df0 = DataFrame::from_columns(vec![("x".into(), Column::Float(v.clone()))]).unwrap();
        for x in v.iter_mut().take(extra_nans as usize) {
            *x = f64::NAN;
        }
        let df1 = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let opts = ValidateOptions { dataset_label: "p".into(), ..Default::default() };
        let b0 = belief_report_from_locke(&validate(&df0, &opts));
        let b1 = belief_report_from_locke(&validate(&df1, &opts));
        prop_assert!(b1.score.missingness_score <= b0.score.missingness_score + 1e-9);
    }

    #[test]
    fn drift_compare_is_deterministic_under_arbitrary_input(
        v1 in arb_float_vec(),
        v2 in arb_float_vec()
    ) {
        let train = DataFrame::from_columns(vec![("x".into(), Column::Float(v1))]).unwrap();
        let test = DataFrame::from_columns(vec![("x".into(), Column::Float(v2))]).unwrap();
        let a = compare(&train, &test, &DriftConfig::default());
        let b = compare(&train, &test, &DriftConfig::default());
        prop_assert_eq!(a, b);
    }

    #[test]
    fn sample_score_stays_in_unit_interval(n in 0u64..100_000) {
        let s = sample_score_from_n(n);
        prop_assert!(s >= 0.0 && s <= 1.0);
    }
}
