// Phase 10 — test_tidy_filter_chain_mask_merge
// Chained filter().filter() must equal a single pass with A && B semantics.
// No column buffers are allocated during chaining.
use cjc_data::{Column, DataFrame, DBinOp, DExpr};

fn make_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        ("y".into(), Column::Float(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])),
    ])
    .unwrap()
}

fn pred_gt(col: &str, val: i64) -> DExpr {
    DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col(col.into())),
        right: Box::new(DExpr::LitInt(val)),
    }
}

fn pred_le(col: &str, val: i64) -> DExpr {
    DExpr::BinOp {
        op: DBinOp::Le,
        left: Box::new(DExpr::Col(col.into())),
        right: Box::new(DExpr::LitInt(val)),
    }
}

#[test]
fn test_tidy_filter_chain_mask_merge() {
    let df = make_df();

    // Chained: x > 2 then x <= 6  → rows where x ∈ {3,4,5,6}
    let chained = df
        .tidy()
        .filter(&pred_gt("x", 2))
        .unwrap()
        .filter(&pred_le("x", 6))
        .unwrap();

    let chained_rows = chained.nrows();
    let chained_mat = chained.materialize().unwrap();

    // Single-pass equivalent
    let df2 = make_df();
    let single = df2
        .tidy()
        .filter(&DExpr::BinOp {
            op: DBinOp::And,
            left: Box::new(pred_gt("x", 2)),
            right: Box::new(pred_le("x", 6)),
        })
        .unwrap();

    let single_rows = single.nrows();
    let single_mat = single.materialize().unwrap();

    assert_eq!(chained_rows, single_rows, "row counts must match");
    assert_eq!(chained_rows, 4); // x=3,4,5,6

    // Verify identical column data
    let chained_x = chained_mat.get_column("x").unwrap();
    let single_x = single_mat.get_column("x").unwrap();
    if let (cjc_data::Column::Int(cv), cjc_data::Column::Int(sv)) = (chained_x, single_x) {
        assert_eq!(cv, sv, "chained and single-pass must produce identical rows");
        assert_eq!(*cv, vec![3i64, 4, 5, 6]);
    } else {
        panic!("expected Int columns");
    }
}

#[test]
fn test_tidy_filter_chain_no_materialization() {
    // Chaining two filters must not allocate column buffers — only bitmask words.
    // We verify by checking the underlying mask rather than materializing.
    let df = make_df();
    let v1 = df.tidy().filter(&pred_gt("x", 4)).unwrap(); // x in {5,6,7,8}
    let v2 = v1.filter(&pred_le("x", 6)).unwrap();         // x in {5,6}

    // Mask count (not nrows of materialized df) should be 2
    assert_eq!(v2.mask().count_ones(), 2);
    // Projection still identity (all cols)
    assert_eq!(v2.ncols(), 2);
}

#[test]
fn test_tidy_filter_chain_three_levels() {
    let df = make_df();
    // x > 1, x < 8, x != 4  → x in {2,3,5,6,7}
    let v = df
        .tidy()
        .filter(&pred_gt("x", 1))
        .unwrap()
        .filter(&DExpr::BinOp {
            op: DBinOp::Lt,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(8)),
        })
        .unwrap()
        .filter(&DExpr::BinOp {
            op: DBinOp::Ne,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(4)),
        })
        .unwrap();

    assert_eq!(v.nrows(), 5);
    let mat = v.materialize().unwrap();
    if let cjc_data::Column::Int(vals) = mat.get_column("x").unwrap() {
        assert_eq!(*vals, vec![2i64, 3, 5, 6, 7]);
    }
}

#[test]
fn test_tidy_filter_chain_all_masked_out() {
    let df = make_df();
    // Contradictory: x > 5 AND x < 3 → 0 rows
    let v = df
        .tidy()
        .filter(&pred_gt("x", 5))
        .unwrap()
        .filter(&DExpr::BinOp {
            op: DBinOp::Lt,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(3)),
        })
        .unwrap();
    assert_eq!(v.nrows(), 0);
}
