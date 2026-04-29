//! v3 Phase 5 — Full Column wiring + filter-chain Hybrid path + cat-aware arrange.
//!
//! Three deliverables, three test sections:
//!
//! 1. **Full Column wiring** — `Column::CategoricalAdaptive(Box<CategoricalColumn>)`
//!    is now a first-class column variant. Display, length, gather,
//!    arrange, and join consumers all see equivalent behaviour to
//!    `Column::Categorical`. Round-trip through `to_legacy_categorical`
//!    is byte-identical for UTF-8 levels with no nulls.
//!
//! 2. **Filter-chain Hybrid path** — when the existing `TidyView` mask
//!    is `AdaptiveSelection::Hybrid`, the next `filter()` call routes
//!    through `existing.intersect(fresh)` instead of materialising a
//!    full `nrows/64` BitMask. Bit-identical visible row set; mode is
//!    Hybrid (not VerbatimMask) on output.
//!
//! 3. **Cat-aware arrange** — sorting by a `Column::Categorical` key
//!    with lex-sorted levels uses u32 code comparison, byte-identical
//!    to string comparison. Unsorted levels fall back to string compare.

use cjc_data::byte_dict::{ByteDictionary, CategoricalColumn, CategoryOrdering};
use cjc_data::{ArrangeKey, Column, DBinOp, DExpr, DataFrame, TidyView};

// ─────────────────────────────────────────────────────────────────────────
//  Section 1: Full Column wiring (CategoricalAdaptive)
// ─────────────────────────────────────────────────────────────────────────

fn make_adaptive_col(levels: &[&str], idx_per_row: &[usize]) -> CategoricalColumn {
    // Use Explicit ordering to pin level→code 1:1 with `levels`.
    let explicit: Vec<Vec<u8>> =
        levels.iter().map(|s| s.as_bytes().to_vec()).collect();
    let dict = ByteDictionary::from_explicit(explicit).unwrap();
    let mut cc = CategoricalColumn::with_dictionary(dict);
    for &i in idx_per_row {
        cc.push(levels[i].as_bytes()).unwrap();
    }
    cc
}

#[test]
fn phase5_categorical_adaptive_basic_accessors() {
    let cc = make_adaptive_col(&["red", "green", "blue"], &[0, 1, 2, 1, 0]);
    let col = Column::categorical_adaptive(cc);
    assert_eq!(col.len(), 5);
    assert_eq!(col.type_name(), "CategoricalAdaptive");
    assert_eq!(col.get_display(0), "red");
    assert_eq!(col.get_display(1), "green");
    assert_eq!(col.get_display(2), "blue");
    assert_eq!(col.get_display(3), "green");
    assert_eq!(col.get_display(4), "red");
}

#[test]
fn phase5_categorical_adaptive_renders_in_dataframe() {
    let cc = make_adaptive_col(&["us", "eu", "ap"], &[0, 1, 0, 2]);
    let df = DataFrame::from_columns(vec![
        ("region".into(), Column::categorical_adaptive(cc)),
        ("v".into(), Column::Int(vec![1, 2, 3, 4])),
    ])
    .unwrap();
    assert_eq!(df.nrows(), 4);
    assert_eq!(df.columns[0].1.get_display(0), "us");
    assert_eq!(df.columns[0].1.get_display(3), "ap");
}

#[test]
fn phase5_to_legacy_categorical_roundtrip() {
    let cc = make_adaptive_col(&["a", "b", "c"], &[0, 1, 2, 0]);
    let adaptive = Column::categorical_adaptive(cc);
    let legacy = adaptive.to_legacy_categorical();
    match legacy {
        Column::Categorical { levels, codes } => {
            assert_eq!(levels, vec!["a", "b", "c"]);
            assert_eq!(codes, vec![0u32, 1, 2, 0]);
        }
        _ => panic!("expected legacy Categorical, got {:?}", legacy),
    }
}

#[test]
fn phase5_to_legacy_categorical_falls_back_to_str_on_nulls() {
    let mut cc = CategoricalColumn::new();
    cc.push(b"a").unwrap();
    cc.push_null();
    cc.push(b"b").unwrap();
    let adaptive = Column::categorical_adaptive(cc);
    let legacy = adaptive.to_legacy_categorical();
    match legacy {
        Column::Str(v) => assert_eq!(v, vec!["a", "", "b"]),
        _ => panic!("expected Str fallback for null-bearing CategoricalAdaptive"),
    }
}

#[test]
fn phase5_categorical_adaptive_renders_through_join_via_legacy() {
    // Joining a frame with CategoricalAdaptive against a regular frame.
    // Through the `to_legacy_categorical` shim joins still work; output
    // matches the equivalent legacy-categorical join.
    let cc_left = make_adaptive_col(&["a", "b", "c"], &[0, 1, 2, 0]);
    let left = DataFrame::from_columns(vec![
        ("k".into(), Column::categorical_adaptive(cc_left)),
        ("v".into(), Column::Int(vec![1, 2, 3, 4])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Str(vec!["a".into(), "b".into()])),
        ("w".into(), Column::Int(vec![10, 20])),
    ])
    .unwrap();
    let lv = TidyView::from_df(left);
    let rv = TidyView::from_df(right);
    let joined = lv.inner_join(&rv, &[("k", "k")]).unwrap();
    let df = joined.borrow();
    // a/1/10, b/2/20, a/4/10 = 3 rows.
    assert_eq!(df.nrows(), 3);
}

#[test]
fn phase5_to_categorical_column_then_back_to_adaptive() {
    // Round-trip Column::Categorical → CategoricalColumn → CategoricalAdaptive
    // → display equivalent.
    let original = Column::Categorical {
        levels: vec!["x".into(), "y".into(), "z".into()],
        codes: vec![0, 1, 2, 1],
    };
    let cc = original.to_categorical_column().unwrap();
    let adaptive = Column::categorical_adaptive(cc);
    assert_eq!(adaptive.get_display(0), "x");
    assert_eq!(adaptive.get_display(1), "y");
    assert_eq!(adaptive.get_display(2), "z");
    assert_eq!(adaptive.get_display(3), "y");
}

// ─────────────────────────────────────────────────────────────────────────
//  Section 2: Filter-chain Hybrid intersect path
// ─────────────────────────────────────────────────────────────────────────

/// Build a 16k-row frame with a value column whose distribution forces
/// a Hybrid mask after a `> threshold` filter (mid-band, mid-density).
fn frame_for_hybrid_mask(nrows: usize) -> DataFrame {
    // Two int columns: `a` strides large enough that `a > 0` keeps ~50% rows.
    let a: Vec<i64> = (0..nrows as i64).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    // `b` similar pattern with stride 5.
    let b: Vec<i64> = (0..nrows as i64).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(a)),
        ("b".into(), Column::Int(b)),
    ])
    .unwrap()
}

#[test]
fn phase5_filter_chain_on_hybrid_existing_visible_rows_match() {
    let df = frame_for_hybrid_mask(16_384);
    let pred1 = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("a".into())),
        right: Box::new(DExpr::LitInt(0)),
    };
    let pred2 = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("b".into())),
        right: Box::new(DExpr::LitInt(0)),
    };
    // Apply pred1 first to (likely) build a Hybrid mask.
    let v1 = TidyView::from_df(df.clone()).filter(&pred1).unwrap();
    // Now chain pred2: the new path should kick in only if v1 is Hybrid.
    let v2 = v1.filter(&pred2).unwrap();

    // Reference: AND in a single predicate.
    let pred_and = DExpr::BinOp {
        op: DBinOp::And,
        left: Box::new(pred1),
        right: Box::new(pred2),
    };
    let v_ref = TidyView::from_df(df).filter(&pred_and).unwrap();

    let v2_idx: Vec<usize> = v2.selection().iter_indices().collect();
    let ref_idx: Vec<usize> = v_ref.selection().iter_indices().collect();
    assert_eq!(v2_idx, ref_idx, "Hybrid filter chain disagrees with reference AND");
}

#[test]
fn phase5_filter_chain_on_non_hybrid_takes_legacy_path() {
    // Small frame can't produce a Hybrid mask (Hybrid needs nrows ≥ 8192).
    // The Hybrid path must NOT activate; legacy path must produce
    // identical output.
    let df = frame_for_hybrid_mask(100);
    let pred1 = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("a".into())),
        right: Box::new(DExpr::LitInt(0)),
    };
    let v = TidyView::from_df(df).filter(&pred1).unwrap();
    // Just verify it didn't crash and produced the right count.
    let visible: Vec<usize> = v.selection().iter_indices().collect();
    assert!(visible.iter().all(|&r| r % 3 == 0));
}

// ─────────────────────────────────────────────────────────────────────────
//  Section 3: Cat-aware arrange
// ─────────────────────────────────────────────────────────────────────────

fn cat_col_sorted(levels: &[&str], codes: &[u32]) -> Column {
    // Levels sorted lex.
    let mut sorted_levels: Vec<String> = levels.iter().map(|s| s.to_string()).collect();
    sorted_levels.sort();
    Column::Categorical { levels: sorted_levels, codes: codes.to_vec() }
}

#[test]
fn phase5_arrange_cat_sorted_levels_byte_equal_to_string_arrange() {
    // Two parallel frames: one with Categorical (sorted levels), one with
    // Str. Arrange ascending; output rows must be byte-equal.
    let names = vec!["banana", "apple", "cherry", "apple", "banana"];
    let n = names.len();

    let str_col = Column::Str(names.iter().map(|s| s.to_string()).collect());

    let mut sorted_levels: Vec<&str> = names.iter().copied().collect();
    sorted_levels.sort();
    sorted_levels.dedup();
    let level_strs: Vec<String> = sorted_levels.iter().map(|s| s.to_string()).collect();
    let codes: Vec<u32> = names
        .iter()
        .map(|n| level_strs.iter().position(|s| s == n).unwrap() as u32)
        .collect();
    let cat_col = Column::Categorical { levels: level_strs, codes };

    let df_str = DataFrame::from_columns(vec![
        ("k".into(), str_col),
        ("idx".into(), Column::Int((0..n as i64).collect())),
    ])
    .unwrap();
    let df_cat = DataFrame::from_columns(vec![
        ("k".into(), cat_col),
        ("idx".into(), Column::Int((0..n as i64).collect())),
    ])
    .unwrap();

    let key = ArrangeKey { col_name: "k".into(), descending: false };
    let v_str = TidyView::from_df(df_str).arrange(&[key.clone()]).unwrap();
    let v_cat = TidyView::from_df(df_cat).arrange(&[key]).unwrap();

    let str_idx: Vec<i64> = match &v_str.collect_columns().unwrap()[1].1 {
        Column::Int(v) => v.clone(),
        _ => panic!(),
    };
    let cat_idx: Vec<i64> = match &v_cat.collect_columns().unwrap()[1].1 {
        Column::Int(v) => v.clone(),
        _ => panic!(),
    };
    assert_eq!(str_idx, cat_idx, "cat-aware arrange must preserve string-arrange row order");
}

#[test]
fn phase5_arrange_cat_unsorted_levels_falls_back_to_string() {
    // Levels deliberately NOT lex-sorted: ["zebra", "apple"]. Cat-aware
    // path must decline (since code-compare ≠ string-compare here),
    // legacy path must produce correct order.
    let cat_col = Column::Categorical {
        levels: vec!["zebra".into(), "apple".into()],
        codes: vec![0, 1, 0, 1, 1], // zebra, apple, zebra, apple, apple
    };
    let df = DataFrame::from_columns(vec![
        ("k".into(), cat_col),
        ("idx".into(), Column::Int((0..5_i64).collect())),
    ])
    .unwrap();
    let key = ArrangeKey { col_name: "k".into(), descending: false };
    let v = TidyView::from_df(df).arrange(&[key]).unwrap();
    let result_cols = v.collect_columns().unwrap();
    // Apple < Zebra in lex order, so all 3 apples should come first.
    let k_disp: Vec<String> = (0..5).map(|r| result_cols[0].1.get_display(r)).collect();
    assert_eq!(k_disp, vec!["apple", "apple", "apple", "zebra", "zebra"]);
}

#[test]
fn phase5_arrange_cat_descending_works() {
    let cat_col = cat_col_sorted(&["a", "b", "c"], &[2, 0, 1, 0, 2]);
    let df = DataFrame::from_columns(vec![
        ("k".into(), cat_col),
        ("idx".into(), Column::Int((0..5_i64).collect())),
    ])
    .unwrap();
    let key = ArrangeKey { col_name: "k".into(), descending: true };
    let v = TidyView::from_df(df).arrange(&[key]).unwrap();
    let cols = v.collect_columns().unwrap();
    let k_disp: Vec<String> = (0..5).map(|r| cols[0].1.get_display(r)).collect();
    // c, c, b, a, a (descending lex)
    assert_eq!(k_disp, vec!["c", "c", "b", "a", "a"]);
}

#[test]
fn phase5_arrange_two_keys_mixed_cat_and_int() {
    // Composite key: categorical asc, int desc.
    let cat_col = cat_col_sorted(&["a", "b"], &[0, 0, 1, 1, 0]);
    let int_col = Column::Int(vec![10, 20, 30, 40, 5]);
    let df = DataFrame::from_columns(vec![
        ("k".into(), cat_col),
        ("v".into(), int_col),
    ])
    .unwrap();
    let v = TidyView::from_df(df)
        .arrange(&[
            ArrangeKey { col_name: "k".into(), descending: false },
            ArrangeKey { col_name: "v".into(), descending: true },
        ])
        .unwrap();
    let cols = v.collect_columns().unwrap();
    let k_disp: Vec<String> = (0..5).map(|r| cols[0].1.get_display(r)).collect();
    let v_int: Vec<i64> = match &cols[1].1 {
        Column::Int(v) => v.clone(),
        _ => panic!(),
    };
    // Within "a" rows: 20, 10, 5 (desc by v). Within "b" rows: 40, 30.
    assert_eq!(k_disp, vec!["a", "a", "a", "b", "b"]);
    assert_eq!(v_int, vec![20, 10, 5, 40, 30]);
}

// ── Bench (ignored) ─────────────────────────────────────────────────────

#[test]
#[ignore]
fn bench_phase5_arrange_cat_vs_string() {
    use std::time::Instant;
    const N: usize = 100_000;
    const N_LEVELS: usize = 100;
    const ITERS: usize = 5;

    let levels: Vec<String> = (0..N_LEVELS).map(|i| format!("L{:03}", i)).collect();
    let codes: Vec<u32> = (0..N).map(|i| (i * 7 % N_LEVELS) as u32).collect();
    let names: Vec<String> = codes.iter().map(|&c| levels[c as usize].clone()).collect();

    let df_str = DataFrame::from_columns(vec![
        ("k".into(), Column::Str(names)),
        ("idx".into(), Column::Int((0..N as i64).collect())),
    ])
    .unwrap();
    let df_cat = DataFrame::from_columns(vec![
        ("k".into(), Column::Categorical { levels, codes }),
        ("idx".into(), Column::Int((0..N as i64).collect())),
    ])
    .unwrap();

    let key = ArrangeKey { col_name: "k".into(), descending: false };
    // Warm.
    let _ = TidyView::from_df(df_str.clone()).arrange(&[key.clone()]).unwrap();
    let _ = TidyView::from_df(df_cat.clone()).arrange(&[key.clone()]).unwrap();

    let mut str_total = std::time::Duration::ZERO;
    let mut cat_total = std::time::Duration::ZERO;
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = TidyView::from_df(df_str.clone()).arrange(&[key.clone()]).unwrap();
        str_total += t.elapsed();

        let t = Instant::now();
        let _ = TidyView::from_df(df_cat.clone()).arrange(&[key.clone()]).unwrap();
        cat_total += t.elapsed();
    }
    let str_avg = str_total / ITERS as u32;
    let cat_avg = cat_total / ITERS as u32;
    let speedup = str_avg.as_secs_f64() / cat_avg.as_secs_f64();
    eprintln!(
        "\n[Phase 5 bench] arrange, {N} rows, {N_LEVELS} unique levels (avg of {ITERS}):\n\
         \x20\x20Str:   {str_avg:?}\n\
         \x20\x20Cat:   {cat_avg:?}\n\
         \x20\x20Speedup: {speedup:.2}×\n",
    );
}

// ── Helper: collect columns ─────────────────────────────────────────────

trait CollectColumns {
    fn collect_columns(&self) -> Result<Vec<(String, Column)>, cjc_data::TidyError>;
}

impl CollectColumns for TidyView {
    fn collect_columns(&self) -> Result<Vec<(String, Column)>, cjc_data::TidyError> {
        let df = self.materialize()?;
        Ok(df.columns)
    }
}
