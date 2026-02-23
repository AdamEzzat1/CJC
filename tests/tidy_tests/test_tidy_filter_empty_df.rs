// Phase 10 — test_tidy_filter_empty_df
// Filtering a 0-row DataFrame: must not panic; returns 0-row view.
use cjc_data::{Column, DataFrame, DBinOp, DExpr};

fn empty_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("age".into(), Column::Int(vec![])),
        ("score".into(), Column::Float(vec![])),
    ])
    .unwrap()
}

#[test]
fn test_tidy_filter_empty_df() {
    let df = empty_df();
    assert_eq!(df.nrows(), 0);

    let view = df.tidy();
    assert_eq!(view.nrows(), 0);
    assert_eq!(view.ncols(), 2);

    // Any predicate on 0-row frame: no panic
    let predicate = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("age".into())),
        right: Box::new(DExpr::LitInt(18)),
    };
    let filtered = view.filter(&predicate).unwrap();
    assert_eq!(filtered.nrows(), 0);
    assert_eq!(filtered.ncols(), 2);

    let mat = filtered.materialize().unwrap();
    assert_eq!(mat.nrows(), 0);
    assert_eq!(mat.ncols(), 2);
}

#[test]
fn test_tidy_filter_empty_df_mask_is_valid() {
    let df = empty_df();
    let view = df.tidy();
    // BitMask with 0 rows has 0 words and 0 ones
    assert_eq!(view.mask().nrows(), 0);
    assert_eq!(view.mask().count_ones(), 0);
}

#[test]
fn test_tidy_filter_empty_df_unknown_col_still_errors() {
    let df = empty_df();
    let view = df.tidy();
    let err = view
        .filter(&DExpr::Col("nonexistent".into()))
        .unwrap_err();
    // Should still validate column references even for empty frames
    assert!(
        matches!(err, cjc_data::TidyError::ColumnNotFound(_)),
        "expected ColumnNotFound, got {:?}",
        err
    );
}
