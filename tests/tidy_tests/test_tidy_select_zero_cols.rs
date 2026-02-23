// Phase 10 — test_tidy_select_zero_cols
// Selecting 0 columns: valid empty-column view. Shape is [nrows, 0].
use cjc_data::{Column, DataFrame};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 2, 3, 4, 5])),
        ("y".into(), Column::Float(vec![0.1, 0.2, 0.3, 0.4, 0.5])),
    ])
    .unwrap()
}

#[test]
fn test_tidy_select_zero_cols() {
    let df = make_df();
    let view = df.tidy();

    // Select no columns
    let selected = view.select(&[]).unwrap();
    assert_eq!(selected.ncols(), 0, "expected 0 projected columns");
    assert_eq!(selected.nrows(), 5, "row count must be preserved");
    assert_eq!(selected.column_names(), Vec::<&str>::new());
}

#[test]
fn test_tidy_select_zero_cols_materialize() {
    let df = make_df();
    let selected = df.tidy().select(&[]).unwrap();
    let mat = selected.materialize().unwrap();
    assert_eq!(mat.ncols(), 0);
    assert_eq!(mat.nrows(), 0, "no columns => nrows() = 0 (no first column to derive from)");
}

#[test]
fn test_tidy_select_zero_cols_empty_df() {
    let df = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![])),
    ])
    .unwrap();
    let selected = df.tidy().select(&[]).unwrap();
    assert_eq!(selected.ncols(), 0);
    assert_eq!(selected.nrows(), 0);
    let mat = selected.materialize().unwrap();
    assert_eq!(mat.ncols(), 0);
}
