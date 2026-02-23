// Phase 13: bind_rows and bind_cols edge-case tests

use cjc_data::{Column, DataFrame, TidyError};

fn make_ab() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![1, 2, 3])),
        ("b".into(), Column::Float(vec![1.0, 2.0, 3.0])),
    ])
    .unwrap()
}

fn make_ab2() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![4, 5])),
        ("b".into(), Column::Float(vec![4.0, 5.0])),
    ])
    .unwrap()
}

// ── bind_rows basic ───────────────────────────────────────────────────────

#[test]
fn test_bind_rows_row_count() {
    let df1 = make_ab();
    let df2 = make_ab2();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let bound = v1.bind_rows(&v2).unwrap();
    let b = bound.borrow();
    assert_eq!(b.nrows(), 5); // 3 + 2
}

#[test]
fn test_bind_rows_order_left_then_right() {
    let df1 = make_ab();
    let df2 = make_ab2();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let bound = v1.bind_rows(&v2).unwrap();
    let b = bound.borrow();
    if let Column::Int(vals) = b.get_column("a").unwrap() {
        assert_eq!(vals, &[1, 2, 3, 4, 5]);
    } else {
        panic!("expected Int column a");
    }
}

#[test]
fn test_bind_rows_col_order_matches_left() {
    let df1 = make_ab();
    let df2 = make_ab2();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let bound = v1.bind_rows(&v2).unwrap();
    let b = bound.borrow();
    assert_eq!(b.column_names(), vec!["a", "b"]);
}

#[test]
fn test_bind_rows_empty_right() {
    let df1 = make_ab();
    let df2 = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![])),
        ("b".into(), Column::Float(vec![])),
    ])
    .unwrap();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let bound = v1.bind_rows(&v2).unwrap();
    let b = bound.borrow();
    assert_eq!(b.nrows(), 3); // left unchanged
}

#[test]
fn test_bind_rows_empty_left() {
    let df1 = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![])),
        ("b".into(), Column::Float(vec![])),
    ])
    .unwrap();
    let df2 = make_ab2();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let bound = v1.bind_rows(&v2).unwrap();
    let b = bound.borrow();
    assert_eq!(b.nrows(), 2);
}

#[test]
fn test_bind_rows_after_filter() {
    let df1 = make_ab();
    let df2 = make_ab2();
    let v1 = df1.tidy().filter(&cjc_data::DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(cjc_data::DExpr::Col("a".into())),
        right: Box::new(cjc_data::DExpr::LitInt(1)),
    }).unwrap();
    let v2 = df2.tidy();
    let bound = v1.bind_rows(&v2).unwrap();
    let b = bound.borrow();
    assert_eq!(b.nrows(), 4); // 2 (filtered left) + 2 (right)
}

// ── bind_rows error cases ─────────────────────────────────────────────────

#[test]
fn test_bind_rows_schema_mismatch_error() {
    let df1 = make_ab();
    let df2 = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![1])),
        ("c".into(), Column::Float(vec![1.0])), // "c" not "b"
    ])
    .unwrap();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let err = v1.bind_rows(&v2).unwrap_err();
    assert!(matches!(err, TidyError::Internal(_)));
}

#[test]
fn test_bind_rows_type_mismatch_in_col_error() {
    // Same col name, different type
    let df1 = make_ab();
    let df2 = DataFrame::from_columns(vec![
        ("a".into(), Column::Str(vec!["x".into()])), // wrong type
        ("b".into(), Column::Float(vec![1.0])),
    ])
    .unwrap();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let err = v1.bind_rows(&v2).unwrap_err();
    // Either schema mismatch (names differ would catch it) or internal type error
    assert!(err.to_string().len() > 0);
}

// ── bind_cols basic ───────────────────────────────────────────────────────

#[test]
fn test_bind_cols_col_count() {
    let df1 = make_ab();
    let df2 = DataFrame::from_columns(vec![
        ("c".into(), Column::Int(vec![10, 20, 30])),
    ])
    .unwrap();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let bound = v1.bind_cols(&v2).unwrap();
    let b = bound.borrow();
    assert_eq!(b.ncols(), 3); // a, b, c
}

#[test]
fn test_bind_cols_col_order_left_then_right() {
    let df1 = make_ab();
    let df2 = DataFrame::from_columns(vec![
        ("c".into(), Column::Int(vec![10, 20, 30])),
    ])
    .unwrap();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let bound = v1.bind_cols(&v2).unwrap();
    let b = bound.borrow();
    assert_eq!(b.column_names(), vec!["a", "b", "c"]);
}

#[test]
fn test_bind_cols_values_correct() {
    let df1 = make_ab();
    let df2 = DataFrame::from_columns(vec![
        ("c".into(), Column::Int(vec![10, 20, 30])),
    ])
    .unwrap();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let bound = v1.bind_cols(&v2).unwrap();
    let b = bound.borrow();
    if let Column::Int(c) = b.get_column("c").unwrap() {
        assert_eq!(c, &[10, 20, 30]);
    }
}

#[test]
fn test_bind_cols_row_mismatch_error() {
    let df1 = make_ab();       // 3 rows
    let df2 = make_ab2();      // 2 rows
    let v1 = df1.tidy().select(&["a"]).unwrap();
    let v2 = df2.tidy().select(&["a"]).unwrap();
    // Rename to avoid collision
    let v2r = v2.rename(&[("a", "x")]).unwrap();
    let err = v1.bind_cols(&v2r).unwrap_err();
    assert!(matches!(err, TidyError::LengthMismatch { .. }));
}

#[test]
fn test_bind_cols_name_collision_error() {
    let df1 = make_ab();
    let df2 = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![10, 20, 30])), // "a" collides with left
    ])
    .unwrap();
    let v1 = df1.tidy();
    let v2 = df2.tidy();
    let err = v1.bind_cols(&v2).unwrap_err();
    assert!(matches!(err, TidyError::DuplicateColumn(_)));
}
