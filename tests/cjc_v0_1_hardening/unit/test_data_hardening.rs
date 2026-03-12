//! Data module (cjc-data) hardening tests.
//! Tests the DataFrame API through direct Rust calls.

use cjc_data::{DataFrame, Column};

/// Create an empty DataFrame.
#[test]
fn dataframe_empty() {
    let df = DataFrame::new();
    assert_eq!(df.nrows(), 0);
    assert_eq!(df.ncols(), 0);
}

/// Create a DataFrame with one column.
#[test]
fn dataframe_single_column() {
    let cols = vec![
        ("x".to_string(), Column::Float(vec![1.0, 2.0, 3.0])),
    ];
    let df = DataFrame::from_columns(cols).expect("should create df");
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 1);
}

/// Create a DataFrame with multiple columns.
#[test]
fn dataframe_multi_column() {
    let cols = vec![
        ("x".to_string(), Column::Float(vec![1.0, 2.0, 3.0])),
        ("y".to_string(), Column::Float(vec![4.0, 5.0, 6.0])),
    ];
    let df = DataFrame::from_columns(cols).expect("should create df");
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 2);
}

/// Column access by name via columns vec.
#[test]
fn dataframe_column_exists() {
    let cols = vec![
        ("x".to_string(), Column::Float(vec![1.0, 2.0, 3.0])),
    ];
    let df = DataFrame::from_columns(cols).expect("should create df");
    let has_x = df.columns.iter().any(|(name, _)| name == "x");
    assert!(has_x, "Column 'x' should exist");
}

/// Column access for non-existent name.
#[test]
fn dataframe_column_missing() {
    let cols = vec![
        ("x".to_string(), Column::Float(vec![1.0])),
    ];
    let df = DataFrame::from_columns(cols).expect("should create df");
    let has_y = df.columns.iter().any(|(name, _)| name == "y");
    assert!(!has_y, "Column 'y' should not exist");
}

/// Int column.
#[test]
fn dataframe_int_column() {
    let cols = vec![
        ("ids".to_string(), Column::Int(vec![1, 2, 3])),
    ];
    let df = DataFrame::from_columns(cols).expect("should create df");
    assert_eq!(df.nrows(), 3);
}

/// Bool column.
#[test]
fn dataframe_bool_column() {
    let cols = vec![
        ("flags".to_string(), Column::Bool(vec![true, false, true])),
    ];
    let df = DataFrame::from_columns(cols).expect("should create df");
    assert_eq!(df.nrows(), 3);
}

/// String column.
#[test]
fn dataframe_str_column() {
    let cols = vec![
        ("names".to_string(), Column::Str(vec!["a".into(), "b".into(), "c".into()])),
    ];
    let df = DataFrame::from_columns(cols).expect("should create df");
    assert_eq!(df.nrows(), 3);
}
