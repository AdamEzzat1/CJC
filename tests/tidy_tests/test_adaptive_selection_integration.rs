// Adaptive TidyView Engine v2 — integration tests
//
// Confirms that filter() routes results through the adaptive classifier
// and that downstream consumers (count, materialize, joins, set ops) see
// identical results regardless of the chosen mode.

use cjc_data::{Column, DBinOp, DExpr, DataFrame};

fn pred_gt_int(col: &str, val: i64) -> DExpr {
    DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col(col.into())),
        right: Box::new(DExpr::LitInt(val)),
    }
}

fn pred_lt_int(col: &str, val: i64) -> DExpr {
    DExpr::BinOp {
        op: DBinOp::Lt,
        left: Box::new(DExpr::Col(col.into())),
        right: Box::new(DExpr::LitInt(val)),
    }
}

fn make_int_df(n: usize) -> DataFrame {
    let xs: Vec<i64> = (0..n as i64).collect();
    DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap()
}

// ── Mode classification through the real filter pipeline ─────────────────────

#[test]
fn filter_with_no_match_picks_empty_arm() {
    let df = make_int_df(10_000);
    // No row has x < 0 in [0, 10000)
    let v = df.tidy().filter(&pred_lt_int("x", 0)).unwrap();
    assert_eq!(v.explain_selection_mode(), "Empty");
    assert_eq!(v.nrows(), 0);
}

#[test]
fn filter_matching_every_row_picks_all_arm() {
    let df = make_int_df(10_000);
    // Every row has x >= 0
    let v = df.tidy().filter(&pred_gt_int("x", -1)).unwrap();
    assert_eq!(v.explain_selection_mode(), "All");
    assert_eq!(v.nrows(), 10_000);
}

#[test]
fn filter_with_sparse_matches_picks_selection_vector() {
    // 100k rows, only x > 99_990 → 9 hits → 9 < 100_000/1024 (=97) → sparse
    let df = make_int_df(100_000);
    let v = df.tidy().filter(&pred_gt_int("x", 99_990)).unwrap();
    assert_eq!(v.explain_selection_mode(), "SelectionVector");
    assert_eq!(v.nrows(), 9);
}

#[test]
fn filter_with_dense_matches_picks_verbatim_mask() {
    // 1000 rows, x > 100 → 899 hits → ~90% → dense
    let df = make_int_df(1000);
    let v = df.tidy().filter(&pred_gt_int("x", 100)).unwrap();
    assert_eq!(v.explain_selection_mode(), "VerbatimMask");
    assert_eq!(v.nrows(), 899);
}

// ── Materialization round-trip across modes ──────────────────────────────────

#[test]
fn materialize_agrees_across_modes() {
    let cases = [
        (10_000usize, -1i64, "All", 10_000usize),
        (10_000, 10_000, "Empty", 0),
        (100_000, 99_990, "SelectionVector", 9),
        (1000, 100, "VerbatimMask", 899),
    ];
    for (n, threshold, expected_mode, expected_count) in cases {
        let df = make_int_df(n);
        let v = df.tidy().filter(&pred_gt_int("x", threshold)).unwrap();
        assert_eq!(
            v.explain_selection_mode(),
            expected_mode,
            "mode mismatch for n={n} threshold={threshold}"
        );
        let mat = v.materialize().unwrap();
        assert_eq!(
            mat.nrows(),
            expected_count,
            "materialize row count mismatch in {expected_mode} mode"
        );
    }
}

// ── Chained filters preserve adaptive mode picks ─────────────────────────────

#[test]
fn chained_filters_reclassify_density() {
    let df = make_int_df(100_000);
    // Step 1: dense (most rows pass) → VerbatimMask
    let v1 = df.tidy().filter(&pred_gt_int("x", 100)).unwrap();
    assert_eq!(v1.explain_selection_mode(), "VerbatimMask");
    // Step 2: tight upper bound → very sparse → SelectionVector
    let v2 = v1.filter(&pred_lt_int("x", 110)).unwrap();
    assert_eq!(v2.explain_selection_mode(), "SelectionVector");
    assert_eq!(v2.nrows(), 9); // x in [101, 109]
}

// ── selection() vs mask() inspection accessors agree ─────────────────────────

#[test]
fn selection_and_mask_inspection_agree() {
    let df = make_int_df(100_000);
    let v = df.tidy().filter(&pred_gt_int("x", 99_000)).unwrap();
    let bm = v.mask();
    let sel = v.selection();
    assert_eq!(bm.count_ones(), sel.count());
    assert_eq!(bm.nrows(), sel.nrows());
    let bm_idx: Vec<usize> = bm.iter_set().collect();
    let sel_idx: Vec<usize> = sel.iter_indices().collect();
    assert_eq!(bm_idx, sel_idx);
}
