// CJC Test Suite — DataFrame Inspection Methods
// Tests for head(), tail(), describe(), shape(), columns(), dtypes(), glimpse()
// These test the tidy_dispatch layer directly via Rust API.

use std::rc::Rc;
use cjc_data::{Column, DataFrame, TidyView};
use cjc_data::tidy_dispatch::dispatch_tidy_method;
use cjc_runtime::value::Value;
use std::any::Any;

fn make_test_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("name".into(), Column::Str(vec![
            "alice".into(), "bob".into(), "carol".into(), "dave".into(), "eve".into(),
        ])),
        ("age".into(), Column::Int(vec![25, 30, 35, 40, 45])),
        ("score".into(), Column::Float(vec![88.5, 92.0, 75.3, 81.7, 96.2])),
        ("active".into(), Column::Bool(vec![true, false, true, true, false])),
    ]).unwrap()
}

fn wrap_view(view: TidyView) -> Rc<dyn Any> {
    Rc::new(view) as Rc<dyn Any>
}

// ── shape() ─────────────────────────────────────────────────────────────

#[test]
fn test_shape_returns_tuple() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "shape", &[]).unwrap().unwrap();
    match result {
        Value::Tuple(t) => {
            assert_eq!(t.len(), 2);
            assert!(matches!(&t[0], Value::Int(5)));  // 5 rows
            assert!(matches!(&t[1], Value::Int(4)));  // 4 columns
        }
        _ => panic!("shape should return Tuple"),
    }
}

// ── columns() ───────────────────────────────────────────────────────────

#[test]
fn test_columns_returns_names() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "columns", &[]).unwrap().unwrap();
    match result {
        Value::Array(a) => {
            let names: Vec<String> = a.iter().map(|v| format!("{}", v)).collect();
            assert_eq!(names, vec!["name", "age", "score", "active"]);
        }
        _ => panic!("columns should return Array"),
    }
}

// ── dtypes() ────────────────────────────────────────────────────────────

#[test]
fn test_dtypes_returns_struct() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "dtypes", &[]).unwrap().unwrap();
    match result {
        Value::Struct { name, fields } => {
            assert_eq!(name, "Dtypes");
            assert_eq!(format!("{}", fields.get("name").unwrap()), "Str");
            assert_eq!(format!("{}", fields.get("age").unwrap()), "Int");
            assert_eq!(format!("{}", fields.get("score").unwrap()), "Float");
            assert_eq!(format!("{}", fields.get("active").unwrap()), "Bool");
        }
        _ => panic!("dtypes should return Struct"),
    }
}

// ── head() ──────────────────────────────────────────────────────────────

#[test]
fn test_head_default() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    // Default head shows up to 10 rows (we have 5)
    let result = dispatch_tidy_method(&view, "head", &[]).unwrap().unwrap();
    match result {
        Value::String(s) => {
            assert!(s.contains("alice"), "head should show first row");
            assert!(s.contains("eve"), "head should show all 5 rows (< 10 default)");
        }
        _ => panic!("head should return String"),
    }
}

#[test]
fn test_head_with_n() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "head", &[Value::Int(2)]).unwrap().unwrap();
    match result {
        Value::String(s) => {
            assert!(s.contains("alice"), "head(2) should show first row");
            assert!(s.contains("bob"), "head(2) should show second row");
            assert!(!s.contains("carol"), "head(2) should NOT show third row");
        }
        _ => panic!("head should return String"),
    }
}

// ── tail() ──────────────────────────────────────────────────────────────

#[test]
fn test_tail_with_n() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "tail", &[Value::Int(2)]).unwrap().unwrap();
    match result {
        Value::String(s) => {
            assert!(!s.contains("alice"), "tail(2) should NOT show first row");
            assert!(s.contains("dave"), "tail(2) should show fourth row");
            assert!(s.contains("eve"), "tail(2) should show last row");
        }
        _ => panic!("tail should return String"),
    }
}

// ── describe() ──────────────────────────────────────────────────────────

#[test]
fn test_describe_includes_all_columns() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "describe", &[]).unwrap().unwrap();
    match result {
        Value::String(s) => {
            assert!(s.contains("── name (Str) ──"), "describe should show name column header");
            assert!(s.contains("── age (Int) ──"), "describe should show age column header");
            assert!(s.contains("── score (Float) ──"), "describe should show score column header");
            assert!(s.contains("── active (Bool) ──"), "describe should show active column header");
        }
        _ => panic!("describe should return String"),
    }
}

#[test]
fn test_describe_numeric_stats() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "describe", &[]).unwrap().unwrap();
    match result {
        Value::String(s) => {
            assert!(s.contains("count: 5"), "describe should show count for numeric columns");
            assert!(s.contains("mean:"), "describe should show mean");
            assert!(s.contains("std:"), "describe should show std");
            assert!(s.contains("min:"), "describe should show min");
            assert!(s.contains("max:"), "describe should show max");
        }
        _ => panic!("describe should return String"),
    }
}

#[test]
fn test_describe_string_stats() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "describe", &[]).unwrap().unwrap();
    match result {
        Value::String(s) => {
            assert!(s.contains("unique: 5"), "describe should show unique count for Str column");
            assert!(s.contains("top:"), "describe should show most frequent value");
        }
        _ => panic!("describe should return String"),
    }
}

// ── glimpse() ───────────────────────────────────────────────────────────

#[test]
fn test_glimpse_format() {
    let df = make_test_df();
    let view = wrap_view(df.tidy());
    let result = dispatch_tidy_method(&view, "glimpse", &[]).unwrap().unwrap();
    match result {
        Value::String(s) => {
            assert!(s.contains("Rows: 5"), "glimpse should show row count, got: {}", s);
            assert!(s.contains("Columns: 4"), "glimpse should show column count, got: {}", s);
            assert!(s.contains("name"), "glimpse should list column names, got: {}", s);
            // Check type markers
            assert!(s.contains("Str"), "glimpse should show Str type, got: {}", s);
            assert!(s.contains("Int"), "glimpse should show Int type, got: {}", s);
            assert!(s.contains("alice"), "glimpse should show preview values, got: {}", s);
        }
        _ => panic!("glimpse should return String"),
    }
}

// ── Determinism ─────────────────────────────────────────────────────────

#[test]
fn test_describe_determinism() {
    let df = make_test_df();
    let view1 = wrap_view(df.tidy());
    let r1 = dispatch_tidy_method(&view1, "describe", &[]).unwrap().unwrap();

    let df2 = make_test_df();
    let view2 = wrap_view(df2.tidy());
    let r2 = dispatch_tidy_method(&view2, "describe", &[]).unwrap().unwrap();

    assert_eq!(format!("{}", r1), format!("{}", r2), "describe must be deterministic");
}

#[test]
fn test_glimpse_determinism() {
    let df = make_test_df();
    let view1 = wrap_view(df.tidy());
    let r1 = dispatch_tidy_method(&view1, "glimpse", &[]).unwrap().unwrap();

    let df2 = make_test_df();
    let view2 = wrap_view(df2.tidy());
    let r2 = dispatch_tidy_method(&view2, "glimpse", &[]).unwrap().unwrap();

    assert_eq!(format!("{}", r1), format!("{}", r2), "glimpse must be deterministic");
}
