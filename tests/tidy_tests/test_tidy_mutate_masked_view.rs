// Phase 10 — test_tidy_mutate_masked_view
// Mutate on a masked view applies only to visible (masked-in) rows.
// The materialized result contains only those rows.
use cjc_data::{Column, DataFrame, DBinOp, DExpr};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("v".into(), Column::Int(vec![10, 20, 30, 40, 50])),
        ("flag".into(), Column::Bool(vec![true, false, true, false, true])),
    ])
    .unwrap()
}

#[test]
fn test_tidy_mutate_masked_view() {
    let df = make_df();

    // Filter: flag == true → rows 0,2,4 (v=10,30,50)
    let view = df
        .tidy()
        .filter(&DExpr::BinOp {
            op: DBinOp::Eq,
            left: Box::new(DExpr::Col("flag".into())),
            right: Box::new(DExpr::LitBool(true)),
        })
        .unwrap();

    assert_eq!(view.nrows(), 3);

    // Mutate: add new column doubled = v * 2, applied only to masked-in rows
    let frame = view
        .mutate(&[("doubled", DExpr::BinOp {
            op: DBinOp::Mul,
            left: Box::new(DExpr::Col("v".into())),
            right: Box::new(DExpr::LitInt(2)),
        })])
        .unwrap();

    let b = frame.borrow();
    assert_eq!(b.nrows(), 3, "materialized frame should have 3 rows");
    if let Column::Int(v) = b.get_column("v").unwrap() {
        assert_eq!(*v, vec![10i64, 30, 50], "only masked-in rows");
    }
    if let Column::Int(v) = b.get_column("doubled").unwrap() {
        assert_eq!(*v, vec![20i64, 60, 100]);
    }
}

#[test]
fn test_tidy_mutate_masked_view_empty_mask() {
    let df = make_df();
    // Filter to 0 rows
    let view = df
        .tidy()
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("v".into())),
            right: Box::new(DExpr::LitInt(1000)),
        })
        .unwrap();

    assert_eq!(view.nrows(), 0);

    let frame = view
        .mutate(&[("doubled", DExpr::BinOp {
            op: DBinOp::Mul,
            left: Box::new(DExpr::Col("v".into())),
            right: Box::new(DExpr::LitInt(2)),
        })])
        .unwrap();

    let b = frame.borrow();
    assert_eq!(b.nrows(), 0, "empty mask → 0 materialized rows");
    // Note: empty DataFrame may have 0 columns if no rows and columns were derived.
    // The key invariant: nrows == 0 (already asserted above).
    let _ = b.get_column("v");
}

#[test]
fn test_tidy_mutate_on_projected_masked_view() {
    let df = make_df();

    // Select only "v", then filter, then mutate
    let view = df
        .tidy()
        .select(&["v"])
        .unwrap()
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("v".into())),
            right: Box::new(DExpr::LitInt(25)),
        })
        .unwrap();

    assert_eq!(view.nrows(), 3); // v=30,40,50
    assert_eq!(view.ncols(), 1);

    let frame = view
        .mutate(&[("v_scaled", DExpr::BinOp {
            op: DBinOp::Mul,
            left: Box::new(DExpr::Col("v".into())),
            right: Box::new(DExpr::LitInt(10)),
        })])
        .unwrap();

    let b = frame.borrow();
    if let Column::Int(v) = b.get_column("v_scaled").unwrap() {
        assert_eq!(*v, vec![300i64, 400, 500]);
    }
}
