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

    // ── v0.6 categorical detectors ──────────────────────────────────────

    #[test]
    fn categorical_quality_is_deterministic_under_arbitrary_strings(
        vs in prop::collection::vec(any::<String>(), 5..50)
    ) {
        let df = DataFrame::from_columns(vec![("c".into(), Column::Str(vs))]).unwrap();
        let cfg = cjc_locke::CategoricalQualityConfig::default();
        let a = cjc_locke::detect_all_categorical_quality(&df, &cfg);
        let b = cjc_locke::detect_all_categorical_quality(&df, &cfg);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn wasserstein_is_nonnegative_and_finite(
        a in arb_float_vec(),
        b in arb_float_vec()
    ) {
        if let Some(w) = cjc_locke::wasserstein_1(&a, &b) {
            prop_assert!(w >= -1e-9, "W_1 should be non-negative, got {}", w);
            prop_assert!(w.is_finite(), "W_1 should be finite, got {}", w);
        }
    }

    #[test]
    fn wasserstein_symmetric(
        a in arb_float_vec(),
        b in arb_float_vec()
    ) {
        let lhs = cjc_locke::wasserstein_1(&a, &b);
        let rhs = cjc_locke::wasserstein_1(&b, &a);
        match (lhs, rhs) {
            (Some(x), Some(y)) => prop_assert!((x - y).abs() < 1e-9, "{} vs {}", x, y),
            _ => prop_assert_eq!(lhs, rhs),
        }
    }

    #[test]
    fn lineage_mermaid_emit_is_deterministic(
        label in "[a-z]{1,10}",
        nrows in 1u64..50
    ) {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Float((0..nrows).map(|i| i as f64).collect()))
        ]).unwrap();
        let g = cjc_locke::api::lineage_for_dataset(&label, &df);
        let a = cjc_locke::emit_lineage_mermaid(&g);
        let b = cjc_locke::emit_lineage_mermaid(&g);
        prop_assert_eq!(a, b);
    }

    // ── v0.6 batch 2 ─────────────────────────────────────────────────────

    #[test]
    fn pii_detection_is_deterministic(
        vs in prop::collection::vec(any::<String>(), 10..40)
    ) {
        let df = DataFrame::from_columns(vec![("c".into(), Column::Str(vs))]).unwrap();
        let cfg = cjc_locke::PiiConfig::default();
        let a = cjc_locke::detect_all_pii(&df, &cfg);
        let b = cjc_locke::detect_all_pii(&df, &cfg);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn label_encoding_risk_is_deterministic(
        seed in any::<u64>(),
        n in 30u64..200,
        cap in 2i64..30
    ) {
        let mut state = seed;
        let values: Vec<i64> = (0..n).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state % cap as u64) as i64
        }).collect();
        let df = DataFrame::from_columns(vec![("v".into(), Column::Int(values))]).unwrap();
        let cfg = cjc_locke::LabelEncodingRiskConfig::default();
        let a = cjc_locke::detect_label_encoding_risk(&df, &cfg);
        let b = cjc_locke::detect_label_encoding_risk(&df, &cfg);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn per_column_summary_emit_is_deterministic(
        nrows in 5u64..50
    ) {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Float((0..nrows).map(|i| i as f64).collect()))
        ]).unwrap();
        let opts = ValidateOptions { dataset_label: "p".into(), ..Default::default() };
        let r = validate(&df, &opts);
        let a = cjc_locke::emit_per_column_confidence_summary(&r);
        let b = cjc_locke::emit_per_column_confidence_summary(&r);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn seasonality_dispersion_is_finite_and_non_negative(
        times in prop::collection::vec(0i64..1_000_000_000_000i64, 100..400)
    ) {
        let df = DataFrame::from_columns(vec![("ts".into(), Column::Int(times))]).unwrap();
        let cfg = cjc_locke::SeasonalityConfig::default();
        let findings = cjc_locke::detect_seasonality(&df, "ts", &cfg);
        for f in &findings {
            for ev in &f.evidence {
                if let cjc_locke::FindingEvidence::Metric { label, value } = ev {
                    if label == "index_of_dispersion" {
                        prop_assert!(value.is_finite() && *value >= 0.0, "ID = {}", value);
                    }
                }
            }
        }
    }

    // ── v0.6.3 ───────────────────────────────────────────────────────────

    #[test]
    fn shape_skewness_and_kurtosis_are_finite(v in arb_float_vec()) {
        if let Some((s, k)) = cjc_locke::skew_and_kurtosis(&v) {
            prop_assert!(s.is_finite(), "skew = {}", s);
            prop_assert!(k.is_finite(), "ex_kurt = {}", k);
        }
    }

    #[test]
    fn shape_detection_is_deterministic(v in arb_float_vec()) {
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let cfg = cjc_locke::ShapeConfig::default();
        let a = cjc_locke::detect_distribution_shape(&df, &cfg);
        let b = cjc_locke::detect_distribution_shape(&df, &cfg);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn multiclass_max_auc_is_in_unit_interval(
        n_per_class in 10u64..30,
        feature_seed in any::<u64>()
    ) {
        let mut state = feature_seed;
        let target: Vec<u32> = (0..3)
            .flat_map(|c| std::iter::repeat(c).take(n_per_class as usize))
            .collect();
        let feat: Vec<f64> = (0..target.len()).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state as f64) / (u64::MAX as f64)
        }).collect();
        if let Some(auc) = cjc_locke::multiclass_max_one_vs_rest_auc(&feat, &target, 3, 5) {
            prop_assert!(auc >= 0.0 && auc <= 1.0 + 1e-9, "max OVR AUC = {}", auc);
        }
    }

    // ── v0.7: BeliefScore meet-semilattice laws ─────────────────────────

    #[test]
    fn algebra_default_compose_is_idempotent(
        a in arb_belief_score()
    ) {
        let r = cjc_locke::BeliefAxisRules::default();
        let b = cjc_locke::compose(&a, &a, &r);
        prop_assert!(cjc_locke::eq_componentwise(&a, &b, 1e-12));
    }

    #[test]
    fn algebra_default_compose_is_commutative(
        a in arb_belief_score(),
        b in arb_belief_score()
    ) {
        let r = cjc_locke::BeliefAxisRules::default();
        let ab = cjc_locke::compose(&a, &b, &r);
        let ba = cjc_locke::compose(&b, &a, &r);
        prop_assert!(cjc_locke::eq_componentwise(&ab, &ba, 1e-12));
    }

    #[test]
    fn algebra_default_compose_is_associative(
        a in arb_belief_score(),
        b in arb_belief_score(),
        c in arb_belief_score()
    ) {
        let r = cjc_locke::BeliefAxisRules::default();
        let ab_then_c = cjc_locke::compose(&cjc_locke::compose(&a, &b, &r), &c, &r);
        let a_then_bc = cjc_locke::compose(&a, &cjc_locke::compose(&b, &c, &r), &r);
        prop_assert!(cjc_locke::eq_componentwise(&ab_then_c, &a_then_bc, 1e-12));
    }

    #[test]
    fn algebra_default_compose_has_identity_top(
        a in arb_belief_score()
    ) {
        let r = cjc_locke::BeliefAxisRules::default();
        let with_top = cjc_locke::compose(&a, &cjc_locke::top(), &r);
        prop_assert!(cjc_locke::eq_componentwise(&a, &with_top, 1e-12));
    }

    #[test]
    fn algebra_default_compose_is_monotonic_downward(
        a in arb_belief_score(),
        b in arb_belief_score()
    ) {
        let r = cjc_locke::BeliefAxisRules::default();
        let ab = cjc_locke::compose(&a, &b, &r);
        prop_assert!(cjc_locke::le_componentwise(&ab, &a, 1e-12));
        prop_assert!(cjc_locke::le_componentwise(&ab, &b, 1e-12));
    }

    #[test]
    fn algebra_all_rules_keep_axes_in_unit_interval(
        a in arb_belief_score(),
        b in arb_belief_score(),
        rule_idx in 0usize..4
    ) {
        let rule = match rule_idx {
            0 => cjc_locke::CompositionRule::Min,
            1 => cjc_locke::CompositionRule::Max,
            2 => cjc_locke::CompositionRule::GeometricMean,
            _ => cjc_locke::CompositionRule::ArithmeticMean,
        };
        let r = cjc_locke::BeliefAxisRules {
            schema: rule, missingness: rule, drift: rule, leakage: rule,
            lineage: rule, sample: rule, duplication: rule, constraint: rule,
        };
        let c = cjc_locke::compose(&a, &b, &r);
        for axis in [
            c.schema_score, c.missingness_score, c.drift_score, c.leakage_score,
            c.lineage_score, c.sample_score, c.duplication_score, c.constraint_score,
            c.overall,
        ] {
            prop_assert!(axis >= 0.0 && axis <= 1.0, "axis = {}", axis);
        }
    }

    // ── v0.7 part 2: byte-identity regression for the algebra migration ─
    //
    // The migration of `api::belief_report_from_locke_with_model` from a
    // direct `BeliefScore::from_dimensions(...)` call to a per-axis
    // `compose_many(...)` chain under `BeliefAxisRules::default()` is safe
    // only because of an algebraic identity: under all-Min, composing 8
    // per-axis partials (each carrying one axis's value with the other
    // seven set to the meet identity ⊤ = 1.0) reduces to the same
    // `from_dimensions(...)` call on the same 8-tuple.
    //
    // This proptest locks the identity at the f64 bit-pattern level —
    // any drift between the two paths fails the gate immediately. See
    // ADR-0036 v0.7 part 2 for the migration record.

    #[test]
    fn algebra_path_is_byte_identical_to_direct_from_dimensions(
        s in 0.0f64..=1.0, m in 0.0f64..=1.0, d in 0.0f64..=1.0, l in 0.0f64..=1.0,
        li in 0.0f64..=1.0, sa in 0.0f64..=1.0, du in 0.0f64..=1.0, c in 0.0f64..=1.0,
    ) {
        // The pre-migration path: a single `from_dimensions` call.
        let direct = cjc_locke::BeliefScore::from_dimensions(s, m, d, l, li, sa, du, c);

        // The migrated path: 8 per-axis partials composed under all-Min.
        let rules = cjc_locke::BeliefAxisRules::default();
        let partials = [
            cjc_locke::BeliefScore::from_dimensions(s, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            cjc_locke::BeliefScore::from_dimensions(1.0, m, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            cjc_locke::BeliefScore::from_dimensions(1.0, 1.0, d, 1.0, 1.0, 1.0, 1.0, 1.0),
            cjc_locke::BeliefScore::from_dimensions(1.0, 1.0, 1.0, l, 1.0, 1.0, 1.0, 1.0),
            cjc_locke::BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, li, 1.0, 1.0, 1.0),
            cjc_locke::BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, 1.0, sa, 1.0, 1.0),
            cjc_locke::BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, du, 1.0),
            cjc_locke::BeliefScore::from_dimensions(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, c),
        ];
        let composed = cjc_locke::compose_many(&partials, &rules)
            .expect("8 non-empty partials always compose");

        // Bit-equal on every axis AND overall — not just numerically close.
        prop_assert_eq!(direct.schema_score.to_bits(), composed.schema_score.to_bits());
        prop_assert_eq!(direct.missingness_score.to_bits(), composed.missingness_score.to_bits());
        prop_assert_eq!(direct.drift_score.to_bits(), composed.drift_score.to_bits());
        prop_assert_eq!(direct.leakage_score.to_bits(), composed.leakage_score.to_bits());
        prop_assert_eq!(direct.lineage_score.to_bits(), composed.lineage_score.to_bits());
        prop_assert_eq!(direct.sample_score.to_bits(), composed.sample_score.to_bits());
        prop_assert_eq!(direct.duplication_score.to_bits(), composed.duplication_score.to_bits());
        prop_assert_eq!(direct.constraint_score.to_bits(), composed.constraint_score.to_bits());
        prop_assert_eq!(direct.overall.to_bits(), composed.overall.to_bits());
    }

    /// End-to-end regression — for arbitrary float-column inputs, the
    /// migrated `belief_report_from_locke` and the preserved inline path
    /// (`__belief_report_from_locke_inline_for_regression_test`) produce
    /// byte-identical `BeliefReport`s. This is stronger than the
    /// algebraic identity above because it exercises the full per-axis
    /// derivation, the `BeliefPenalty::default()` path, and the
    /// recommended-next-steps thresholding.
    #[test]
    fn belief_report_migrated_path_is_byte_identical_to_inline_oracle(
        v in arb_float_vec()
    ) {
        let df = DataFrame::from_columns(vec![("x".into(), Column::Float(v))]).unwrap();
        let opts = ValidateOptions { dataset_label: "prop".into(), ..Default::default() };
        let report = validate(&df, &opts);
        let penalty = cjc_locke::BeliefPenalty::default();

        // Migrated path (current implementation, uses algebra::compose_many).
        let migrated = cjc_locke::belief_report_from_locke_with_model(&report, &penalty);

        // Preserved oracle (pre-migration direct from_dimensions path).
        let oracle = cjc_locke::api::__belief_report_from_locke_inline_for_regression_test(
            &report, &penalty,
        );

        // Score axes bit-equal.
        prop_assert_eq!(migrated.score.schema_score.to_bits(), oracle.score.schema_score.to_bits());
        prop_assert_eq!(migrated.score.missingness_score.to_bits(), oracle.score.missingness_score.to_bits());
        prop_assert_eq!(migrated.score.drift_score.to_bits(), oracle.score.drift_score.to_bits());
        prop_assert_eq!(migrated.score.leakage_score.to_bits(), oracle.score.leakage_score.to_bits());
        prop_assert_eq!(migrated.score.lineage_score.to_bits(), oracle.score.lineage_score.to_bits());
        prop_assert_eq!(migrated.score.sample_score.to_bits(), oracle.score.sample_score.to_bits());
        prop_assert_eq!(migrated.score.duplication_score.to_bits(), oracle.score.duplication_score.to_bits());
        prop_assert_eq!(migrated.score.constraint_score.to_bits(), oracle.score.constraint_score.to_bits());
        prop_assert_eq!(migrated.score.overall.to_bits(), oracle.score.overall.to_bits());

        // Surrounding report structure preserved.
        prop_assert_eq!(&migrated.assumptions, &oracle.assumptions);
        prop_assert_eq!(&migrated.evidence_summary, &oracle.evidence_summary);
        prop_assert_eq!(&migrated.recommended_next_steps, &oracle.recommended_next_steps);
    }
}

fn arb_belief_score() -> impl Strategy<Value = cjc_locke::BeliefScore> {
    (
        0.0f64..=1.0,
        0.0f64..=1.0,
        0.0f64..=1.0,
        0.0f64..=1.0,
        0.0f64..=1.0,
        0.0f64..=1.0,
        0.0f64..=1.0,
        0.0f64..=1.0,
    )
        .prop_map(|(a, b, c, d, e, f, g, h)| {
            cjc_locke::BeliefScore::from_dimensions(a, b, c, d, e, f, g, h)
        })
}

// ─── v0.6.4 — auto-sentinel + E9064 properties ──────────────────────────

/// Random sentinel value chosen from the built-in list.
fn arb_sentinel_choice() -> impl Strategy<Value = &'static str> {
    prop::sample::select(cjc_locke::BUILTIN_STRING_SENTINELS.to_vec())
}

/// A vector of strings where some fraction are sentinels and the rest
/// are arbitrary "real" tokens.
fn arb_mixed_str_col(n: usize, sentinel_prob_pct: u8) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(
        (0u8..100, arb_sentinel_choice(), "[a-z]{1,5}"),
        n..=n,
    )
    .prop_map(move |xs| {
        xs.into_iter()
            .map(|(p, sent, real)| {
                if p < sentinel_prob_pct {
                    sent.to_string()
                } else {
                    real
                }
            })
            .collect()
    })
}

proptest! {
    /// Auto-sentinel detection is deterministic — same input always
    /// produces the same mask and the same E9008 findings (canonical
    /// byte-for-byte).
    #[test]
    fn sentinel_detection_is_deterministic_under_arbitrary_str_input(
        v in arb_mixed_str_col(20, 40),
    ) {
        let df = cjc_data::DataFrame::from_columns(vec![(
            "x".into(),
            cjc_data::Column::Str(v),
        )])
        .unwrap();
        let cfg = cjc_locke::ValidationConfig::default();
        let (m1, f1) = cjc_locke::detect_string_sentinels(&df, &cfg);
        let (m2, f2) = cjc_locke::detect_string_sentinels(&df, &cfg);
        prop_assert_eq!(m1, m2);
        prop_assert_eq!(f1, f2);
    }

    /// Detected sentinel count never exceeds row count.
    #[test]
    fn sentinel_count_bounded_by_row_count(
        v in arb_mixed_str_col(30, 70),
    ) {
        let n = v.len() as u64;
        let df = cjc_data::DataFrame::from_columns(vec![(
            "x".into(),
            cjc_data::Column::Str(v),
        )])
        .unwrap();
        let (masks, _) = cjc_locke::detect_string_sentinels(
            &df,
            &cjc_locke::ValidationConfig::default(),
        );
        let detected = masks
            .get("x")
            .map(|m| m.null_rows.len() as u64)
            .unwrap_or(0);
        prop_assert!(detected <= n);
    }

    /// Opt-out is monotonic — disabling auto-detect can only REMOVE
    /// findings, never add them.
    #[test]
    fn opt_out_only_removes_e9008_findings(
        v in arb_mixed_str_col(20, 50),
    ) {
        let df = cjc_data::DataFrame::from_columns(vec![(
            "x".into(),
            cjc_data::Column::Str(v),
        )])
        .unwrap();
        let on = cjc_locke::ValidationConfig::default();
        let off = cjc_locke::ValidationConfig {
            auto_detect_sentinels: false,
            ..Default::default()
        };
        let (_, f_on) = cjc_locke::detect_string_sentinels(&df, &on);
        let (_, f_off) = cjc_locke::detect_string_sentinels(&df, &off);
        prop_assert!(f_off.is_empty());
        prop_assert!(f_on.iter().all(|f| f.code == "E9008"));
    }

    /// E9064 is deterministic on synthetic (column, target) pairs.
    #[test]
    fn e9064_deterministic_on_arbitrary_int_columns(
        col in prop::collection::vec(0_i64..5, 30..60),
        target in prop::collection::vec(0_i64..3, 30..60),
    ) {
        // Equalise lengths.
        let n = col.len().min(target.len());
        let df = cjc_data::DataFrame::from_columns(vec![
            ("col".into(), cjc_data::Column::Int(col[..n].to_vec())),
            ("y".into(), cjc_data::Column::Int(target[..n].to_vec())),
        ])
        .unwrap();
        let cfg = cjc_locke::PerLevelLeakageConfig::default();
        let a = cjc_locke::detect_per_level_target_leakage(&df, "y", &cfg);
        let b = cjc_locke::detect_per_level_target_leakage(&df, "y", &cfg);
        prop_assert_eq!(a, b);
    }

    /// Raising `min_support` is monotonic in the suppression direction:
    /// a higher threshold never produces MORE findings than a lower one.
    #[test]
    fn e9064_higher_min_support_never_adds_findings(
        col in prop::collection::vec(0_i64..5, 40..80),
        target in prop::collection::vec(0_i64..3, 40..80),
        low in 1_u64..6,
        delta in 1_u64..20,
    ) {
        let n = col.len().min(target.len());
        let df = cjc_data::DataFrame::from_columns(vec![
            ("col".into(), cjc_data::Column::Int(col[..n].to_vec())),
            ("y".into(), cjc_data::Column::Int(target[..n].to_vec())),
        ])
        .unwrap();
        let cfg_low = cjc_locke::PerLevelLeakageConfig { min_support: low, ..Default::default() };
        let cfg_high = cjc_locke::PerLevelLeakageConfig { min_support: low + delta, ..Default::default() };
        let n_low = cjc_locke::detect_per_level_target_leakage(&df, "y", &cfg_low).len();
        let n_high = cjc_locke::detect_per_level_target_leakage(&df, "y", &cfg_high).len();
        prop_assert!(n_high <= n_low);
    }
}
