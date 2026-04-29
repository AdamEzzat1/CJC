// Adaptive TidyView Engine v2 — determinism gate
//
// The adaptive classifier must produce bit-identical output across:
//   1. repeated runs of the same predicate against the same DataFrame
//   2. equivalent inputs that take different code paths (sparse vs. dense
//      mode landing on the same materialized result)
//
// Determinism rules (CJC-Lang):
//   - SplitMix64 RNG (not used here, but the test follows the convention)
//   - No HashMap/HashSet
//   - Stable iteration order in every selection arm
//   - No FMA, no rayon

use cjc_data::{Column, DBinOp, DExpr, DataFrame};

fn pred_gt(col: &str, v: i64) -> DExpr {
    DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col(col.into())),
        right: Box::new(DExpr::LitInt(v)),
    }
}

fn make_df(n: usize) -> DataFrame {
    let xs: Vec<i64> = (0..n as i64).collect();
    DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap()
}

#[test]
fn repeated_filter_produces_identical_indices_sparse() {
    let n = 100_000;
    let mut runs: Vec<Vec<u32>> = Vec::new();
    for _ in 0..10 {
        let df = make_df(n);
        let v = df.tidy().filter(&pred_gt("x", 99_990)).unwrap();
        assert_eq!(v.explain_selection_mode(), "SelectionVector");
        runs.push(v.selection().materialize_indices());
    }
    let first = &runs[0];
    for r in &runs[1..] {
        assert_eq!(first, r, "non-deterministic sparse selection across runs");
    }
}

#[test]
fn repeated_filter_produces_identical_indices_dense() {
    let n = 1000;
    let mut runs: Vec<Vec<u32>> = Vec::new();
    for _ in 0..10 {
        let df = make_df(n);
        let v = df.tidy().filter(&pred_gt("x", 100)).unwrap();
        assert_eq!(v.explain_selection_mode(), "VerbatimMask");
        runs.push(v.selection().materialize_indices());
    }
    let first = &runs[0];
    for r in &runs[1..] {
        assert_eq!(first, r, "non-deterministic dense selection across runs");
    }
}

#[test]
fn repeated_filter_produces_identical_mask_words() {
    // Same predicate, repeat 10×, ALL modes — assert words[] is byte-identical.
    let cases = [(100_000usize, 99_990i64), (1000, 100), (10_000, -1), (10_000, 10_000)];
    for (n, threshold) in cases {
        let mut hashes: Vec<Vec<u64>> = Vec::new();
        for _ in 0..10 {
            let df = make_df(n);
            let v = df.tidy().filter(&pred_gt("x", threshold)).unwrap();
            // Materialize through the BitMask path — words must be bit-equal.
            let bm = v.mask();
            hashes.push(bm.words_slice().to_vec());
        }
        let first = &hashes[0];
        for r in &hashes[1..] {
            assert_eq!(first, r, "non-deterministic words for n={n} threshold={threshold}");
        }
    }
}

#[test]
fn equivalent_predicates_produce_same_indices_across_modes() {
    // The adaptive engine should produce the same final answer regardless
    // of which arm the classifier chose — only the in-memory representation
    // differs. Build the same logical selection two ways and check equality.
    let n = 100_000;
    let df = make_df(n);

    // Path 1: sparse-friendly predicate (only 9 rows pass) → SelectionVector
    let v_sparse = df.tidy().filter(&pred_gt("x", 99_990)).unwrap();
    assert_eq!(v_sparse.explain_selection_mode(), "SelectionVector");
    let idx_sparse = v_sparse.selection().materialize_indices();

    // Path 2: same logical answer reconstructed from the materialized BitMask
    let bm = v_sparse.mask();
    let idx_from_mask: Vec<u32> = bm.iter_set().map(|i| i as u32).collect();

    assert_eq!(
        idx_sparse, idx_from_mask,
        "SelectionVector and BitMask must agree on indices"
    );
}

#[test]
fn iteration_order_is_strictly_ascending_for_every_arm() {
    let cases = [
        (10_000usize, -1i64),     // All
        (10_000, 10_000),         // Empty
        (100_000, 99_990),        // SelectionVector
        (1000, 100),              // VerbatimMask
    ];
    for (n, threshold) in cases {
        let df = make_df(n);
        let v = df.tidy().filter(&pred_gt("x", threshold)).unwrap();
        let collected: Vec<usize> = v.selection().iter_indices().collect();
        for w in collected.windows(2) {
            assert!(
                w[0] < w[1],
                "non-ascending iteration in mode {} (n={n}, threshold={threshold})",
                v.explain_selection_mode()
            );
        }
    }
}
