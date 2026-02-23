// Phase 12 — distinct
// Tests: single col, multi col, zero cols, unknown col, ordering, after projection
use cjc_data::{Column, DataFrame, DBinOp, DExpr, TidyError};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![1, 2, 1, 3, 2, 1])),
        ("b".into(), Column::Str(vec!["x".into(), "y".into(), "x".into(), "z".into(), "y".into(), "w".into()])),
        ("c".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    ])
    .unwrap()
}

#[test]
fn test_distinct_single_col() {
    let df = make_df();
    let view = df.tidy().distinct(&["a"]).unwrap();
    // a: 1,2,1,3,2,1 → distinct first-occurrence: 1(row0), 2(row1), 3(row3)
    assert_eq!(view.nrows(), 3);
    let mat = view.materialize().unwrap();
    if let Column::Int(v) = mat.get_column("a").unwrap() {
        assert_eq!(*v, vec![1i64, 2, 3]);
    }
}

#[test]
fn test_distinct_multi_col() {
    let df = make_df();
    let view = df.tidy().distinct(&["a", "b"]).unwrap();
    // Pairs: (1,x),(2,y),(1,x),(3,z),(2,y),(1,w)
    // Distinct first-occurrence: (1,x)row0, (2,y)row1, (3,z)row3, (1,w)row5
    assert_eq!(view.nrows(), 4);
    let mat = view.materialize().unwrap();
    if let Column::Int(v) = mat.get_column("a").unwrap() {
        assert_eq!(*v, vec![1i64, 2, 3, 1]);
    }
}

#[test]
fn test_distinct_zero_cols_keeps_first_row() {
    let df = make_df();
    let view = df.tidy().distinct(&[]).unwrap();
    // 0 keys → all rows equal on empty key → keep only first row
    assert_eq!(view.nrows(), 1);
}

#[test]
fn test_distinct_unknown_col_errors() {
    let df = make_df();
    let err = df.tidy().distinct(&["nonexistent"]).unwrap_err();
    assert!(matches!(err, TidyError::ColumnNotFound(_)));
}

#[test]
fn test_distinct_ordering_first_occurrence() {
    let df = DataFrame::from_columns(vec![
        ("v".into(), Column::Str(vec!["b".into(), "a".into(), "b".into(), "c".into(), "a".into()])),
    ])
    .unwrap();
    let view = df.tidy().distinct(&["v"]).unwrap();
    let mat = view.materialize().unwrap();
    if let Column::Str(v) = mat.get_column("v").unwrap() {
        // First-occurrence: b, a, c
        assert_eq!(*v, vec!["b".to_string(), "a".to_string(), "c".to_string()]);
    }
}

#[test]
fn test_distinct_empty_df() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![])),
    ])
    .unwrap();
    let view = df.tidy().distinct(&["x"]).unwrap();
    assert_eq!(view.nrows(), 0);
}

#[test]
fn test_distinct_all_unique() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 2, 3, 4, 5])),
    ])
    .unwrap();
    let view = df.tidy().distinct(&["x"]).unwrap();
    assert_eq!(view.nrows(), 5);
}

#[test]
fn test_distinct_all_same() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![7, 7, 7, 7])),
    ])
    .unwrap();
    let view = df.tidy().distinct(&["x"]).unwrap();
    assert_eq!(view.nrows(), 1);
    let mat = view.materialize().unwrap();
    if let Column::Int(v) = mat.get_column("x").unwrap() {
        assert_eq!(*v, vec![7i64]);
    }
}

#[test]
fn test_distinct_after_filter() {
    let df = make_df();
    // Filter to a > 1
    let view = df
        .tidy()
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("a".into())),
            right: Box::new(DExpr::LitInt(1)),
        })
        .unwrap()
        .distinct(&["a"])
        .unwrap();
    // Filtered rows: a=2(row1),3(row3),2(row4) → distinct: 2,3
    assert_eq!(view.nrows(), 2);
    let mat = view.materialize().unwrap();
    if let Column::Int(v) = mat.get_column("a").unwrap() {
        assert_eq!(*v, vec![2i64, 3]);
    }
}

#[test]
fn test_distinct_after_projection() {
    let df = make_df();
    // Select only "a", then distinct
    let view = df
        .tidy()
        .select(&["a"])
        .unwrap()
        .distinct(&["a"])
        .unwrap();
    assert_eq!(view.nrows(), 3);
    assert_eq!(view.ncols(), 1);
}

#[test]
fn test_distinct_to_tensor() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 2, 1, 3])),
        ("y".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0])),
    ])
    .unwrap();
    let view = df.tidy().distinct(&["x"]).unwrap();
    // Distinct x: 1(row0), 2(row1), 3(row3) → y values 1.0, 2.0, 4.0
    let tensor = view.to_tensor(&["x", "y"]).unwrap();
    assert_eq!(tensor.shape(), &[3, 2]);
}
