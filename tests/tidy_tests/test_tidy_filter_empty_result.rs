// Phase 10 — test_tidy_filter_empty_result
// Filter where no rows match: result is a valid 0-row view.
use cjc_data::{Column, DataFrame, DBinOp, DExpr};

fn sample_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 2, 3])),
        ("label".into(), Column::Str(vec!["a".into(), "b".into(), "c".into()])),
    ])
    .unwrap()
}

#[test]
fn test_tidy_filter_empty_result() {
    let df = sample_df();
    let view = df.tidy();

    // Predicate that matches nothing (x > 100)
    let predicate = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("x".into())),
        right: Box::new(DExpr::LitInt(100)),
    };

    let filtered = view.filter(&predicate).unwrap();
    // Zero rows match
    assert_eq!(filtered.nrows(), 0, "expected 0 visible rows");
    // Column count unchanged
    assert_eq!(filtered.ncols(), 2);

    // Materialize: should produce empty-row DataFrame
    let mat = filtered.materialize().unwrap();
    assert_eq!(mat.nrows(), 0);
    assert_eq!(mat.ncols(), 2);

    // Column buffers exist but are empty
    let x_col = mat.get_column("x").unwrap();
    assert_eq!(x_col.len(), 0);
}

#[test]
fn test_tidy_filter_empty_result_column_names_preserved() {
    let df = sample_df();
    let view = df.tidy();
    let filtered = view
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(1000)),
        })
        .unwrap();
    // Column names still stable in order
    assert_eq!(filtered.column_names(), vec!["x", "label"]);
}
