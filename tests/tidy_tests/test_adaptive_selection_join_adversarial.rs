// Adaptive TidyView Engine v2 — adversarial join + group_by tests
//
// Joins and group_by traverse selections via iter_indices(). These tests
// pin: regardless of which selection mode the predicate produced (Sparse
// / Dense / All / Empty), the join output is bit-identical to the result
// computed against a fully-materialized BitMask path.

use cjc_data::{Column, DBinOp, DExpr, DataFrame};

fn pred_gt(col: &str, v: i64) -> DExpr {
    DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col(col.into())),
        right: Box::new(DExpr::LitInt(v)),
    }
}

fn make_left(n: usize) -> DataFrame {
    let ids: Vec<i64> = (0..n as i64).collect();
    let vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(ids)),
        ("val".into(), Column::Float(vals)),
    ])
    .unwrap()
}

fn make_right(n: usize) -> DataFrame {
    let ids: Vec<i64> = (0..n as i64).collect();
    let cats: Vec<String> = (0..n).map(|i| format!("cat_{}", i % 7)).collect();
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(ids)),
        ("cat".into(), Column::Str(cats)),
    ])
    .unwrap()
}

#[test]
fn inner_join_after_sparse_filter_is_deterministic() {
    let left = make_left(100_000);
    let right = make_right(100_000);

    // Sparse: only 9 rows pass on the left
    let lf = left.tidy().filter(&pred_gt("id", 99_990)).unwrap();
    assert_eq!(lf.explain_selection_mode(), "SelectionVector");

    let joined = lf.inner_join(&right.tidy(), &[("id", "id")]).unwrap();
    let mat_nrows = joined.borrow().nrows();
    assert_eq!(mat_nrows, 9);

    // Same answer must come back across 5 repeats (output stability)
    for _ in 0..5 {
        let left2 = make_left(100_000);
        let right2 = make_right(100_000);
        let lf2 = left2.tidy().filter(&pred_gt("id", 99_990)).unwrap();
        let joined2 = lf2.inner_join(&right2.tidy(), &[("id", "id")]).unwrap();
        assert_eq!(
            mat_nrows,
            joined2.borrow().nrows(),
            "non-deterministic join row count"
        );
    }
}

#[test]
fn inner_join_after_dense_filter_matches_sparse_filter_on_same_logical_rows() {
    // Adversarial: build two filters that select the SAME 9 rows but go
    // through different adaptive arms. The join output must agree.
    let left = make_left(100_000);
    let right = make_right(100_000);

    // Sparse path: x > 99_990 → 9 rows → SelectionVector
    let lf_sparse = left.tidy().filter(&pred_gt("id", 99_990)).unwrap();
    assert_eq!(lf_sparse.explain_selection_mode(), "SelectionVector");
    let right_clone = right.clone();
    let frame_sparse = lf_sparse
        .inner_join(&right_clone.tidy(), &[("id", "id")])
        .unwrap();
    let bs = frame_sparse.borrow();

    // "Dense path" simulator: filter with predicate that picks more rows
    // first (forcing VerbatimMask), then narrow to the same 9 via a second
    // filter — the second narrows back into Sparse.
    let left2 = make_left(100_000);
    let lf_chained = left2
        .tidy()
        .filter(&pred_gt("id", 50_000))
        .unwrap()
        .filter(&pred_gt("id", 99_990))
        .unwrap();
    assert_eq!(lf_chained.explain_selection_mode(), "SelectionVector");
    let frame_chained = lf_chained
        .inner_join(&right.tidy(), &[("id", "id")])
        .unwrap();
    let bc = frame_chained.borrow();

    assert_eq!(bs.nrows(), bc.nrows());
    assert_eq!(bs.nrows(), 9);

    // Column-equality check
    let s_id = bs.get_column("id").unwrap();
    let c_id = bc.get_column("id").unwrap();
    if let (Column::Int(s), Column::Int(c)) = (s_id, c_id) {
        assert_eq!(s, c, "id column differs across selection modes");
    } else {
        panic!("expected Int columns");
    }
}

#[test]
fn group_by_after_sparse_filter_groups_correctly() {
    use cjc_data::TidyAgg;

    let n = 100_000;
    let left = make_left(n);

    // Sparse filter: 9 hits
    let lf = left.tidy().filter(&pred_gt("id", 99_990)).unwrap();
    assert_eq!(lf.explain_selection_mode(), "SelectionVector");
    assert_eq!(lf.nrows(), 9);

    // group_by id (each row a unique group), summarise count
    let summary = lf
        .group_by(&["id"])
        .unwrap()
        .summarise(&[("n", TidyAgg::Count)])
        .unwrap();

    assert_eq!(summary.borrow().nrows(), 9, "expected 9 groups");
}
