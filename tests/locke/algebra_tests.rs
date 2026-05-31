//! Integration tests for the v0.7 BeliefScore composition algebra.
//!
//! These tests verify the meet-semilattice laws on belief scores
//! computed from *real validate() runs*, not just hand-built scores.
//! That checks the algebra's contract holds against the actual belief
//! pipeline, not just the algebraic combinators in isolation.

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    api::{belief_report_from_locke, validate, ValidateOptions},
    compose, compose_many, compose_many_arithmetic, compose_weighted,
    eq_componentwise, le_componentwise, top, BeliefAxisRules,
    CompositionRule, ValidationConfig,
};

fn belief_of(df: &DataFrame, label: &str) -> cjc_locke::BeliefScore {
    let opts = ValidateOptions {
        dataset_label: label.into(),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let r = validate(df, &opts);
    belief_report_from_locke(&r).score
}

fn df_clean(n: usize) -> DataFrame {
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    DataFrame::from_columns(vec![("x".into(), Column::Float(xs))]).unwrap()
}

fn df_dirty(n: usize, frac_nan: f64) -> DataFrame {
    let cutoff = (n as f64 * frac_nan) as usize;
    let xs: Vec<f64> = (0..n)
        .map(|i| if i < cutoff { f64::NAN } else { i as f64 })
        .collect();
    DataFrame::from_columns(vec![("x".into(), Column::Float(xs))]).unwrap()
}

#[test]
fn min_compose_is_idempotent_on_real_belief() {
    let r = BeliefAxisRules::default();
    let b = belief_of(&df_clean(100), "clean");
    let c = compose(&b, &b, &r);
    assert!(
        eq_componentwise(&b, &c, 1e-9),
        "idempotence broke on real belief: b={:?}, c={:?}",
        b,
        c
    );
}

#[test]
fn min_compose_with_top_is_identity_on_real_belief() {
    let r = BeliefAxisRules::default();
    let b = belief_of(&df_dirty(100, 0.2), "dirty");
    let c = compose(&b, &top(), &r);
    assert!(eq_componentwise(&b, &c, 1e-9));
}

#[test]
fn min_compose_is_monotonically_downward() {
    let r = BeliefAxisRules::default();
    let clean = belief_of(&df_clean(100), "clean");
    let dirty = belief_of(&df_dirty(100, 0.3), "dirty");
    let merged = compose(&clean, &dirty, &r);
    // merged ≤ clean and merged ≤ dirty (component-wise).
    assert!(le_componentwise(&merged, &clean, 1e-9));
    assert!(le_componentwise(&merged, &dirty, 1e-9));
}

#[test]
fn compose_many_chained_equals_pairwise_on_real_belief() {
    let r = BeliefAxisRules::default();
    let a = belief_of(&df_clean(50), "a");
    let b = belief_of(&df_dirty(50, 0.1), "b");
    let c = belief_of(&df_dirty(50, 0.3), "c");
    let chained = compose(&compose(&a, &b, &r), &c, &r);
    let many = compose_many(&[a, b, c], &r).unwrap();
    assert!(eq_componentwise(&chained, &many, 1e-12));
}

#[test]
fn weighted_compose_lands_between_extremes() {
    // Two leaves: one clean, one very dirty. The weighted average should
    // fall strictly between the two on every axis.
    let clean = belief_of(&df_clean(200), "clean");
    let very_dirty = belief_of(&df_dirty(200, 0.7), "very_dirty");
    let mixed = compose_weighted(&[clean.clone(), very_dirty.clone()], &[1.0, 1.0]).unwrap();

    // The weighted mean lies between the two on every axis.
    for axis in [
        "missingness",
        "constraint",
        "schema",
        "sample",
        "duplication",
    ] {
        let (c, v, m) = match axis {
            "missingness" => (clean.missingness_score, very_dirty.missingness_score, mixed.missingness_score),
            "constraint" => (clean.constraint_score, very_dirty.constraint_score, mixed.constraint_score),
            "schema" => (clean.schema_score, very_dirty.schema_score, mixed.schema_score),
            "sample" => (clean.sample_score, very_dirty.sample_score, mixed.sample_score),
            "duplication" => (clean.duplication_score, very_dirty.duplication_score, mixed.duplication_score),
            _ => unreachable!(),
        };
        let lo = c.min(v);
        let hi = c.max(v);
        assert!(
            m >= lo - 1e-9 && m <= hi + 1e-9,
            "axis {} mean {} not in [{}, {}]",
            axis,
            m,
            lo,
            hi
        );
    }
}

#[test]
fn arithmetic_mean_across_many_leaves_matches_unweighted_mean() {
    let leaves: Vec<cjc_locke::BeliefScore> = (0..5)
        .map(|i| belief_of(&df_dirty(60, 0.05 * i as f64), &format!("leaf{}", i)))
        .collect();
    let mean_all_at_once = compose_many_arithmetic(&leaves).unwrap();
    // The all-at-once mean should equal a single arithmetic mean per axis.
    let n = leaves.len() as f64;
    let avg_schema: f64 = leaves.iter().map(|s| s.schema_score).sum::<f64>() / n;
    assert!((mean_all_at_once.schema_score - avg_schema).abs() < 1e-9);
}

#[test]
fn alternative_rules_keep_axis_values_in_unit_interval() {
    // Boundedness: every variant of CompositionRule produces values in [0, 1]
    // when inputs are in [0, 1].
    let a = belief_of(&df_clean(100), "a");
    let b = belief_of(&df_dirty(100, 0.3), "b");
    for rule in [
        CompositionRule::Min,
        CompositionRule::Max,
        CompositionRule::GeometricMean,
        CompositionRule::ArithmeticMean,
    ] {
        let rules = BeliefAxisRules {
            schema: rule,
            missingness: rule,
            drift: rule,
            leakage: rule,
            lineage: rule,
            sample: rule,
            duplication: rule,
            constraint: rule,
        };
        let c = compose(&a, &b, &rules);
        assert!(
            c.schema_score >= 0.0 && c.schema_score <= 1.0,
            "rule {:?} produced out-of-range schema_score {}",
            rule,
            c.schema_score
        );
        assert!(c.missingness_score >= 0.0 && c.missingness_score <= 1.0);
        assert!(c.constraint_score >= 0.0 && c.constraint_score <= 1.0);
        assert!(c.overall >= 0.0 && c.overall <= 1.0);
    }
}

#[test]
fn compose_is_deterministic_across_runs() {
    let r = BeliefAxisRules::default();
    let a = belief_of(&df_clean(50), "a");
    let b = belief_of(&df_dirty(50, 0.2), "b");
    let r1 = compose(&a, &b, &r);
    let r2 = compose(&a, &b, &r);
    assert_eq!(r1, r2);
}
