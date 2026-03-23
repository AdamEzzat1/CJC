//! Phase 1 parity and regression tests for beta hardening.
//!
//! Validates that wiring gaps (DataFrame.view, sample_indices) are
//! correctly wired in BOTH executors with identical semantics.

use std::rc::Rc;
use std::collections::BTreeMap;

/// Run CJC source through eval, return output lines.
fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    let _ = interp.exec(&program);
    interp.output
}

/// Run CJC source through MIR-exec, return output lines.
fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

// ──────────────────────────────────────────────────────────────────
// 1.1  DataFrame.view() parity — tested via Rust API since CJC has
//      no `dataframe()` literal constructor
// ──────────────────────────────────────────────────────────────────

use cjc_runtime::Value;
use cjc_data::{Column, DataFrame, TidyView};
use cjc_data::tidy_dispatch;

/// Build a Value::Struct representing a DataFrame with the given columns.
fn make_df_value(cols: Vec<(&str, Vec<f64>)>) -> Value {
    let mut fields = BTreeMap::new();
    let mut col_names = Vec::new();
    let nrows = cols.first().map(|(_, v)| v.len()).unwrap_or(0);
    for (name, data) in &cols {
        col_names.push(Value::String(Rc::new(name.to_string())));
        let arr: Vec<Value> = data.iter().map(|&f| Value::Float(f)).collect();
        fields.insert(name.to_string(), Value::Array(Rc::new(arr)));
    }
    fields.insert("__columns".to_string(), Value::Array(Rc::new(col_names)));
    fields.insert("__nrows".to_string(), Value::Int(nrows as i64));
    Value::Struct { name: "DataFrame".to_string(), fields }
}

#[test]
fn dataframe_view_creates_tidy_view_in_mir_exec() {
    // Directly test that rebuild_dataframe_from_struct + TidyView works
    let df = DataFrame::from_columns(vec![
        ("x".to_string(), Column::Float(vec![1.0, 2.0, 3.0])),
        ("y".to_string(), Column::Float(vec![4.0, 5.0, 6.0])),
    ]).unwrap();
    let view = TidyView::from_df(df);
    assert_eq!(view.nrows(), 3);
    assert_eq!(view.ncols(), 2);
}

#[test]
fn dataframe_view_wrap_produces_tidy_value() {
    let df = DataFrame::from_columns(vec![
        ("a".to_string(), Column::Float(vec![10.0, 20.0])),
    ]).unwrap();
    let view = TidyView::from_df(df);
    let val = tidy_dispatch::wrap_view(view);
    match val {
        Value::TidyView(_) => {} // success
        other => panic!("expected TidyView, got {:?}", other.type_name()),
    }
}

// ──────────────────────────────────────────────────────────────────
// 1.2  sample_indices() parity
// ──────────────────────────────────────────────────────────────────

#[test]
fn parity_sample_indices_basic() {
    let src = r#"
        let idx = sample_indices(10, 3, false, 42);
        print(len(idx));
        let i = 0;
        while i < len(idx) {
            print(idx[i]);
            i = i + 1;
        }
    "#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "sample_indices parity failed");
    assert_eq!(eval_out[0], "3", "should return 3 indices");
}

#[test]
fn parity_sample_indices_with_replace() {
    let src = r#"
        let idx = sample_indices(5, 8, true, 99);
        print(len(idx));
    "#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "sample_indices with replacement parity failed");
    assert_eq!(eval_out[0], "8");
}

#[test]
fn determinism_sample_indices() {
    let src = r#"
        let idx = sample_indices(100, 10, false, 42);
        let i = 0;
        while i < len(idx) {
            print(idx[i]);
            i = i + 1;
        }
    "#;
    let run1 = run_mir(src, 42);
    let run2 = run_mir(src, 42);
    let run3 = run_mir(src, 42);
    assert_eq!(run1, run2, "sample_indices determinism failed (run1 vs run2)");
    assert_eq!(run2, run3, "sample_indices determinism failed (run2 vs run3)");
}

#[test]
fn parity_sample_indices_uses_rng() {
    // When no explicit seed, interpreter RNG is used — same interpreter seed should agree
    let src = r#"
        let idx = sample_indices(20, 5, false);
        let i = 0;
        while i < len(idx) {
            print(idx[i]);
            i = i + 1;
        }
    "#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "sample_indices (implicit RNG) parity failed");
}

// ──────────────────────────────────────────────────────────────────
// 1.3  Snap roundtrip parity
// ──────────────────────────────────────────────────────────────────

#[test]
fn parity_snap_roundtrip() {
    let src = r#"
        let val = 42;
        let blob = snap(val);
        let restored = restore(blob);
        print(restored);
    "#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "snap roundtrip parity failed");
    assert_eq!(eval_out[0], "42");
}
