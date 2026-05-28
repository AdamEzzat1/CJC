//! Integration tests for the causal-guardrail module.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::causal_guardrail;
use cjc_locke::causal::{CausalConfig, CausalMode, CausalWarningKind};

fn floats(name: &str, v: Vec<f64>) -> (String, Column) {
    (name.into(), Column::Float(v))
}

#[test]
fn strong_pairwise_correlation_emits_warning() {
    // a and b perfectly anti-correlated.
    let n = 200;
    let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| -(i as f64)).collect();
    let df = DataFrame::from_columns(vec![floats("a", a), floats("b", b)]).unwrap();
    let r = causal_guardrail(&df, None, &CausalConfig::default(), None, false);
    assert!(r.warnings.iter().any(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention));
    assert_eq!(
        r.disclaimer,
        cjc_locke::causal::CausalGuardrailReport::DISCLAIMER
    );
}

#[test]
fn observational_only_mode_emits_extra_warning() {
    let n = 200;
    let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 1.0).collect();
    let df = DataFrame::from_columns(vec![floats("a", a), floats("b", b)]).unwrap();
    let mut cfg = CausalConfig::default();
    cfg.mode = CausalMode::ObservationalOnly;
    let r = causal_guardrail(&df, None, &cfg, None, false);
    assert!(r.warnings.iter().any(|w| w.kind == CausalWarningKind::ObservationalOnly));
}

#[test]
fn no_warnings_for_weakly_correlated_columns() {
    // x and y nearly independent.
    let xs: Vec<f64> = (0..100).map(|i| (i % 7) as f64).collect();
    let ys: Vec<f64> = (0..100).map(|i| (i % 11) as f64).collect();
    let df = DataFrame::from_columns(vec![floats("x", xs), floats("y", ys)]).unwrap();
    let r = causal_guardrail(&df, None, &CausalConfig::default(), None, false);
    assert!(!r.warnings.iter().any(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention));
}

#[test]
fn metadata_with_causal_language_warns() {
    let df = DataFrame::from_columns(vec![floats("x", vec![1.0])]).unwrap();
    let r = causal_guardrail(
        &df,
        None,
        &CausalConfig::default(),
        Some("the model proves that age causes income"),
        false,
    );
    assert!(r.warnings.iter().any(|w| w.kind == CausalWarningKind::CausalLanguageInLabel));
}

#[test]
fn declared_causal_dag_downgrades_strong_correlation_warning() {
    use cjc_locke::CausalDag;
    let n = 200;
    let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 1.0).collect();
    let df = DataFrame::from_columns(vec![floats("a", a), floats("b", b)]).unwrap();
    let mut dag = CausalDag::new();
    dag.add_edge("a", "b").unwrap();
    let mut cfg = CausalConfig::default();
    cfg.assumed_dag = dag;
    let r = causal_guardrail(&df, None, &cfg, None, false);
    let w = r
        .warnings
        .iter()
        .find(|w| w.kind == CausalWarningKind::StrongCorrelationNoIntervention)
        .expect("strong-corr warning emitted");
    assert!(w.message.contains("acknowledged by causal DAG"));
}

#[test]
fn causal_dag_cycle_construction_is_rejected_integration() {
    use cjc_locke::{CausalDag, CausalDagError};
    let mut dag = CausalDag::new();
    dag.add_edge("a", "b").unwrap();
    dag.add_edge("b", "c").unwrap();
    let res = dag.add_edge("c", "a");
    assert!(matches!(res, Err(CausalDagError::CycleIntroduced { .. })));
}

#[test]
fn confounder_hint_fires_with_explicit_target() {
    // age correlates with both income and y; income also correlates with y.
    let n = 200;
    let age: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let income: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0 + 1.0).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5 + 0.5).collect();
    let df = DataFrame::from_columns(vec![floats("age", age), floats("income", income), floats("y", y)]).unwrap();
    let r = causal_guardrail(&df, Some("y"), &CausalConfig::default(), None, false);
    assert!(!r.confounder_hints.is_empty());
}
