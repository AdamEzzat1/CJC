// Phase 13: pivot_wider edge-case tests

use cjc_data::{Column, DataFrame, TidyError};

fn long_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 1, 2, 2])),
        ("name".into(), Column::Str(vec!["a".into(), "b".into(), "a".into(), "b".into()])),
        ("value".into(), Column::Float(vec![10.0, 20.0, 30.0, 40.0])),
    ])
    .unwrap()
}

// ── pivot_wider basic ─────────────────────────────────────────────────────

#[test]
fn test_pivot_wider_basic_schema() {
    let df = long_df();
    let v = df.tidy();
    let wide = v.pivot_wider(&["id"], "name", "value").unwrap();
    let names = wide.column_names();
    // id first, then "a" and "b" in first-occurrence order
    assert_eq!(names, vec!["id", "a", "b"]);
}

#[test]
fn test_pivot_wider_basic_row_count() {
    let df = long_df();
    let v = df.tidy();
    let wide = v.pivot_wider(&["id"], "name", "value").unwrap();
    // 2 unique id values → 2 output rows
    assert_eq!(wide.nrows(), 2);
}

#[test]
fn test_pivot_wider_values_correct() {
    let df = long_df();
    let v = df.tidy();
    let wide = v.pivot_wider(&["id"], "name", "value").unwrap();
    // id=1: a=10.0, b=20.0. id=2: a=30.0, b=40.0
    let col_a = wide.get_column("a").unwrap();
    let col_b = wide.get_column("b").unwrap();
    assert_eq!(col_a.get_display(0), "10");
    assert_eq!(col_b.get_display(0), "20");
    assert_eq!(col_a.get_display(1), "30");
    assert_eq!(col_b.get_display(1), "40");
}

// ── pivot_wider key ordering ──────────────────────────────────────────────

#[test]
fn test_pivot_wider_col_order_first_occurrence() {
    // The key "b" appears before "a" in source → output should be "b", "a"
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 1, 2, 2])),
        ("name".into(), Column::Str(vec!["b".into(), "a".into(), "b".into(), "a".into()])),
        ("value".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0])),
    ])
    .unwrap();
    let v = df.tidy();
    let wide = v.pivot_wider(&["id"], "name", "value").unwrap();
    let names = wide.column_names();
    assert_eq!(names, vec!["id", "b", "a"]);
}

#[test]
fn test_pivot_wider_determinism() {
    let df = long_df();
    let v1 = df.clone().tidy();
    let v2 = df.tidy();
    let w1 = v1.pivot_wider(&["id"], "name", "value").unwrap();
    let w2 = v2.pivot_wider(&["id"], "name", "value").unwrap();
    assert_eq!(w1.column_names(), w2.column_names());
    assert_eq!(w1.nrows(), w2.nrows());
}

// ── pivot_wider missing combinations → null fill ──────────────────────────

#[test]
fn test_pivot_wider_missing_combination_null() {
    // id=1 has only "a", id=2 has only "b" → each missing → null
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2])),
        ("name".into(), Column::Str(vec!["a".into(), "b".into()])),
        ("value".into(), Column::Float(vec![10.0, 20.0])),
    ])
    .unwrap();
    let v = df.tidy();
    let wide = v.pivot_wider(&["id"], "name", "value").unwrap();
    assert_eq!(wide.nrows(), 2);
    let col_a = wide.get_column("a").unwrap();
    let col_b = wide.get_column("b").unwrap();
    // id=1 row: "a" present (10.0), "b" absent (null)
    assert!(!col_a.is_null(0)); // "a" present for id=1
    assert!(col_b.is_null(0));  // "b" absent for id=1
    // id=2 row: "b" present (20.0), "a" absent (null)
    assert!(col_a.is_null(1));  // "a" absent for id=2
    assert!(!col_b.is_null(1)); // "b" present for id=2
}

// ── pivot_wider error cases ───────────────────────────────────────────────

#[test]
fn test_pivot_wider_duplicate_key_strict_error() {
    // id=1 has "a" twice → duplicate key collision
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 1])),
        ("name".into(), Column::Str(vec!["a".into(), "a".into()])),
        ("value".into(), Column::Float(vec![1.0, 2.0])),
    ])
    .unwrap();
    let v = df.tidy();
    let err = v.pivot_wider(&["id"], "name", "value").unwrap_err();
    // Should be a DuplicateColumn (which wraps "duplicate key")
    assert!(matches!(err, TidyError::DuplicateColumn(_)));
}

#[test]
fn test_pivot_wider_unknown_names_from_col() {
    let df = long_df();
    let v = df.tidy();
    let err = v.pivot_wider(&["id"], "nonexistent", "value").unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_pivot_wider_unknown_values_from_col() {
    let df = long_df();
    let v = df.tidy();
    let err = v.pivot_wider(&["id"], "name", "nonexistent").unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_pivot_wider_unknown_id_col() {
    let df = long_df();
    let v = df.tidy();
    let err = v.pivot_wider(&["nonexistent"], "name", "value").unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_pivot_wider_empty_df_zero_rows() {
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![])),
        ("name".into(), Column::Str(vec![])),
        ("value".into(), Column::Float(vec![])),
    ])
    .unwrap();
    let v = df.tidy();
    let wide = v.pivot_wider(&["id"], "name", "value").unwrap();
    assert_eq!(wide.nrows(), 0);
}

#[test]
fn test_pivot_wider_to_tidy_view_filled() {
    let df = long_df();
    let v = df.tidy();
    let wide = v.pivot_wider(&["id"], "name", "value").unwrap();
    let tv = wide.to_tidy_view_filled();
    assert_eq!(tv.nrows(), 2);
    assert_eq!(tv.ncols(), 3); // id, a, b
}
