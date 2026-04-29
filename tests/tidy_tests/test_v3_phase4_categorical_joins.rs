//! v3 Phase 4 — categorical-aware join key path.
//!
//! When every join-key column is `Column::Categorical` on BOTH sides,
//! `inner_join`, `left_join`, `semi_join`, and `anti_join` route through
//! a `Vec<u32>` code-key BTreeMap with a deterministic per-column remap
//! (`right_code → Option<left_code>`). Each DataFrame owns its own
//! dictionary, so the remap is what makes cross-frame comparison safe.
//!
//! These tests use the "shadow frame" technique: build two parallel
//! DataFrames over the same logical strings — one as `Column::Str`
//! (forces the string-key path), one as `Column::Categorical` (triggers
//! the cat-aware fast path) — and assert byte-equal output for every
//! join shape.

use cjc_data::{Column, DBinOp, DExpr, DataFrame, TidyFrame, TidyView};

fn df_string_keys_inner(rows: &[(&str, i64)]) -> DataFrame {
    let names: Vec<String> = rows.iter().map(|(k, _)| (*k).to_string()).collect();
    let vals: Vec<i64> = rows.iter().map(|(_, v)| *v).collect();
    DataFrame::from_columns(vec![
        ("key".to_string(), Column::Str(names)),
        ("v".to_string(), Column::Int(vals)),
    ])
    .unwrap()
}

fn df_categorical_keys_inner(rows: &[(&str, i64)]) -> DataFrame {
    use std::collections::BTreeSet;
    let unique: BTreeSet<&str> = rows.iter().map(|(k, _)| *k).collect();
    let levels: Vec<String> = unique.into_iter().map(String::from).collect();
    let codes: Vec<u32> = rows
        .iter()
        .map(|(k, _)| levels.iter().position(|s| s == k).unwrap() as u32)
        .collect();
    let vals: Vec<i64> = rows.iter().map(|(_, v)| *v).collect();
    DataFrame::from_columns(vec![
        ("key".to_string(), Column::Categorical { levels, codes }),
        ("v".to_string(), Column::Int(vals)),
    ])
    .unwrap()
}

fn df_two_categorical_keys(rows: &[(&str, &str, i64)]) -> DataFrame {
    use std::collections::BTreeSet;
    let region_unique: BTreeSet<&str> = rows.iter().map(|(r, _, _)| *r).collect();
    let bucket_unique: BTreeSet<&str> = rows.iter().map(|(_, b, _)| *b).collect();
    let region_levels: Vec<String> = region_unique.into_iter().map(String::from).collect();
    let bucket_levels: Vec<String> = bucket_unique.into_iter().map(String::from).collect();
    let region_codes: Vec<u32> = rows
        .iter()
        .map(|(r, _, _)| region_levels.iter().position(|s| s == r).unwrap() as u32)
        .collect();
    let bucket_codes: Vec<u32> = rows
        .iter()
        .map(|(_, b, _)| bucket_levels.iter().position(|s| s == b).unwrap() as u32)
        .collect();
    let vals: Vec<i64> = rows.iter().map(|(_, _, v)| *v).collect();
    DataFrame::from_columns(vec![
        (
            "region".into(),
            Column::Categorical { levels: region_levels, codes: region_codes },
        ),
        (
            "bucket".into(),
            Column::Categorical { levels: bucket_levels, codes: bucket_codes },
        ),
        ("v".into(), Column::Int(vals)),
    ])
    .unwrap()
}

fn df_two_string_keys(rows: &[(&str, &str, i64)]) -> DataFrame {
    let regions: Vec<String> = rows.iter().map(|(r, _, _)| (*r).into()).collect();
    let buckets: Vec<String> = rows.iter().map(|(_, b, _)| (*b).into()).collect();
    let vals: Vec<i64> = rows.iter().map(|(_, _, v)| *v).collect();
    DataFrame::from_columns(vec![
        ("region".into(), Column::Str(regions)),
        ("bucket".into(), Column::Str(buckets)),
        ("v".into(), Column::Int(vals)),
    ])
    .unwrap()
}

/// Compare two TidyFrames by extracting column display values per row.
fn frame_rows_display(frame: &TidyFrame) -> Vec<Vec<String>> {
    let df = frame.borrow();
    let n = df.nrows();
    (0..n)
        .map(|r| {
            df.columns
                .iter()
                .map(|(_, col)| col.get_display(r))
                .collect()
        })
        .collect()
}

/// TidyView display rows (for semi/anti, which return views).
fn view_rows_display(view: &TidyView) -> Vec<Vec<String>> {
    let df = view.materialize().unwrap();
    let n = df.nrows();
    (0..n)
        .map(|r| {
            df.columns
                .iter()
                .map(|(_, col)| col.get_display(r))
                .collect()
        })
        .collect()
}

// ── Inner join ──────────────────────────────────────────────────────────

#[test]
fn phase4_inner_join_single_cat_key_matches_string_path() {
    let left_str = df_string_keys_inner(&[("a", 1), ("b", 2), ("c", 3), ("a", 4)]);
    let right_str = df_string_keys_inner(&[("a", 10), ("b", 20), ("d", 40)]);
    let left_cat = df_categorical_keys_inner(&[("a", 1), ("b", 2), ("c", 3), ("a", 4)]);
    let right_cat = df_categorical_keys_inner(&[("a", 10), ("b", 20), ("d", 40)]);

    let lv_str = TidyView::from_df(left_str);
    let rv_str = TidyView::from_df(right_str);
    let lv_cat = TidyView::from_df(left_cat);
    let rv_cat = TidyView::from_df(right_cat);

    let s = lv_str.inner_join(&rv_str, &[("key", "key")]).unwrap();
    let c = lv_cat.inner_join(&rv_cat, &[("key", "key")]).unwrap();
    assert_eq!(frame_rows_display(&s), frame_rows_display(&c));
}

#[test]
fn phase4_inner_join_disjoint_dictionaries() {
    // Right dict has a level "z" not on the left. That right row must
    // never match — the remap returns None and skips it.
    let left = df_categorical_keys_inner(&[("a", 1), ("b", 2)]);
    let right = df_categorical_keys_inner(&[("a", 10), ("z", 999), ("b", 20)]);
    let lv = TidyView::from_df(left);
    let rv = TidyView::from_df(right);
    let joined = lv.inner_join(&rv, &[("key", "key")]).unwrap();
    let rows = frame_rows_display(&joined);
    // Expected: a/1/10 and b/2/20. "z" must not appear.
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r[0] != "z"));
}

#[test]
fn phase4_inner_join_overlapping_disjoint_dictionaries_string_parity() {
    // Stress-test the disjoint-level path against the string-key fallback.
    let left_str = df_string_keys_inner(&[("a", 1), ("b", 2), ("e", 5)]);
    let right_str = df_string_keys_inner(&[("a", 10), ("z", 99), ("b", 20), ("y", 88)]);
    let left_cat = df_categorical_keys_inner(&[("a", 1), ("b", 2), ("e", 5)]);
    let right_cat = df_categorical_keys_inner(&[("a", 10), ("z", 99), ("b", 20), ("y", 88)]);

    let lv_str = TidyView::from_df(left_str);
    let rv_str = TidyView::from_df(right_str);
    let lv_cat = TidyView::from_df(left_cat);
    let rv_cat = TidyView::from_df(right_cat);

    let s = lv_str.inner_join(&rv_str, &[("key", "key")]).unwrap();
    let c = lv_cat.inner_join(&rv_cat, &[("key", "key")]).unwrap();
    assert_eq!(frame_rows_display(&s), frame_rows_display(&c));
}

#[test]
fn phase4_inner_join_two_key_composite() {
    let left_str = df_two_string_keys(&[
        ("us", "hi", 1),
        ("us", "lo", 2),
        ("eu", "hi", 3),
    ]);
    let right_str = df_two_string_keys(&[
        ("us", "hi", 10),
        ("eu", "hi", 30),
        ("eu", "lo", 99),
    ]);
    let left_cat = df_two_categorical_keys(&[
        ("us", "hi", 1),
        ("us", "lo", 2),
        ("eu", "hi", 3),
    ]);
    let right_cat = df_two_categorical_keys(&[
        ("us", "hi", 10),
        ("eu", "hi", 30),
        ("eu", "lo", 99),
    ]);

    let lv_str = TidyView::from_df(left_str);
    let rv_str = TidyView::from_df(right_str);
    let lv_cat = TidyView::from_df(left_cat);
    let rv_cat = TidyView::from_df(right_cat);

    let s = lv_str
        .inner_join(&rv_str, &[("region", "region"), ("bucket", "bucket")])
        .unwrap();
    let c = lv_cat
        .inner_join(&rv_cat, &[("region", "region"), ("bucket", "bucket")])
        .unwrap();
    assert_eq!(frame_rows_display(&s), frame_rows_display(&c));
}

#[test]
fn phase4_inner_join_mixed_keys_falls_back_to_string_path() {
    // One categorical key + one int key → cat-aware path declines, falls
    // back to the string-key path. Output must still be correct.
    let left = DataFrame::from_columns(vec![
        ("k".into(), Column::Categorical {
            levels: vec!["a".into(), "b".into()],
            codes: vec![0, 1, 0],
        }),
        ("n".into(), Column::Int(vec![1, 1, 2])),
    ]).unwrap();
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Categorical {
            levels: vec!["a".into(), "b".into()],
            codes: vec![0, 1],
        }),
        ("n".into(), Column::Int(vec![1, 1])),
    ]).unwrap();
    let lv = TidyView::from_df(left);
    let rv = TidyView::from_df(right);
    let joined = lv.inner_join(&rv, &[("k", "k"), ("n", "n")]).unwrap();
    // Only the rows where both k and n match: (a, 1) and (b, 1).
    assert_eq!(joined.borrow().nrows(), 2);
}

// ── Left join ───────────────────────────────────────────────────────────

#[test]
fn phase4_left_join_preserves_unmatched_left_rows() {
    let left_str = df_string_keys_inner(&[("a", 1), ("z", 99), ("b", 2)]);
    let right_str = df_string_keys_inner(&[("a", 10), ("b", 20)]);
    let left_cat = df_categorical_keys_inner(&[("a", 1), ("z", 99), ("b", 2)]);
    let right_cat = df_categorical_keys_inner(&[("a", 10), ("b", 20)]);

    let lv_str = TidyView::from_df(left_str);
    let rv_str = TidyView::from_df(right_str);
    let lv_cat = TidyView::from_df(left_cat);
    let rv_cat = TidyView::from_df(right_cat);

    let s = lv_str.left_join(&rv_str, &[("key", "key")]).unwrap();
    let c = lv_cat.left_join(&rv_cat, &[("key", "key")]).unwrap();
    assert_eq!(frame_rows_display(&s), frame_rows_display(&c));
}

// ── Semi join ───────────────────────────────────────────────────────────

#[test]
fn phase4_semi_join_categorical_matches_string_path() {
    let left_str = df_string_keys_inner(&[("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    let right_str = df_string_keys_inner(&[("a", 10), ("c", 30)]);
    let left_cat = df_categorical_keys_inner(&[("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    let right_cat = df_categorical_keys_inner(&[("a", 10), ("c", 30)]);

    let lv_str = TidyView::from_df(left_str);
    let rv_str = TidyView::from_df(right_str);
    let lv_cat = TidyView::from_df(left_cat);
    let rv_cat = TidyView::from_df(right_cat);

    let s = lv_str.semi_join(&rv_str, &[("key", "key")]).unwrap();
    let c = lv_cat.semi_join(&rv_cat, &[("key", "key")]).unwrap();
    assert_eq!(view_rows_display(&s), view_rows_display(&c));
}

// ── Anti join ───────────────────────────────────────────────────────────

#[test]
fn phase4_anti_join_categorical_matches_string_path() {
    let left_str = df_string_keys_inner(&[("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    let right_str = df_string_keys_inner(&[("a", 10), ("c", 30)]);
    let left_cat = df_categorical_keys_inner(&[("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    let right_cat = df_categorical_keys_inner(&[("a", 10), ("c", 30)]);

    let lv_str = TidyView::from_df(left_str);
    let rv_str = TidyView::from_df(right_str);
    let lv_cat = TidyView::from_df(left_cat);
    let rv_cat = TidyView::from_df(right_cat);

    let s = lv_str.anti_join(&rv_str, &[("key", "key")]).unwrap();
    let c = lv_cat.anti_join(&rv_cat, &[("key", "key")]).unwrap();
    assert_eq!(view_rows_display(&s), view_rows_display(&c));
}

// ── Determinism ─────────────────────────────────────────────────────────

#[test]
fn phase4_inner_join_categorical_is_deterministic() {
    let left = df_categorical_keys_inner(&[("a", 1), ("b", 2), ("c", 3), ("a", 4)]);
    let right = df_categorical_keys_inner(&[("a", 10), ("b", 20), ("a", 30)]);
    let lv = TidyView::from_df(left);
    let rv = TidyView::from_df(right);
    let r1 = lv.inner_join(&rv, &[("key", "key")]).unwrap();
    let r2 = lv.inner_join(&rv, &[("key", "key")]).unwrap();
    assert_eq!(frame_rows_display(&r1), frame_rows_display(&r2));
}

#[test]
fn phase4_inner_join_categorical_with_filter_pre_visible_rows() {
    // Apply a TidyView filter before the join. The cat-aware path must
    // honour the visible mask exactly the same way the string path does.
    let left = df_categorical_keys_inner(&[
        ("a", 1), ("b", 2), ("c", 3), ("a", 4), ("d", 5)
    ]);
    let right = df_categorical_keys_inner(&[("a", 10), ("b", 20), ("c", 30)]);

    let pred = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("v".into())),
        right: Box::new(DExpr::LitInt(2)),
    };
    let lv = TidyView::from_df(left).filter(&pred).unwrap();
    let rv = TidyView::from_df(right);
    let joined = lv.inner_join(&rv, &[("key", "key")]).unwrap();

    // Same with string keys, parity check.
    let left_s = df_string_keys_inner(&[
        ("a", 1), ("b", 2), ("c", 3), ("a", 4), ("d", 5)
    ]);
    let right_s = df_string_keys_inner(&[("a", 10), ("b", 20), ("c", 30)]);
    let pred_s = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("v".into())),
        right: Box::new(DExpr::LitInt(2)),
    };
    let lv_s = TidyView::from_df(left_s).filter(&pred_s).unwrap();
    let rv_s = TidyView::from_df(right_s);
    let joined_s = lv_s.inner_join(&rv_s, &[("key", "key")]).unwrap();

    assert_eq!(frame_rows_display(&joined), frame_rows_display(&joined_s));
}

// ── Empty edge cases ────────────────────────────────────────────────────

#[test]
fn phase4_inner_join_empty_left_returns_empty() {
    let left = df_categorical_keys_inner(&[]);
    let right = df_categorical_keys_inner(&[("a", 10), ("b", 20)]);
    let lv = TidyView::from_df(left);
    let rv = TidyView::from_df(right);
    let joined = lv.inner_join(&rv, &[("key", "key")]).unwrap();
    assert_eq!(joined.borrow().nrows(), 0);
}

#[test]
fn phase4_inner_join_empty_right_returns_empty() {
    let left = df_categorical_keys_inner(&[("a", 1), ("b", 2)]);
    let right = df_categorical_keys_inner(&[]);
    let lv = TidyView::from_df(left);
    let rv = TidyView::from_df(right);
    let joined = lv.inner_join(&rv, &[("key", "key")]).unwrap();
    assert_eq!(joined.borrow().nrows(), 0);
}

#[test]
fn phase4_left_join_empty_right_returns_all_left_with_nulls() {
    let left = df_categorical_keys_inner(&[("a", 1), ("b", 2)]);
    let right = df_categorical_keys_inner(&[]);
    let lv = TidyView::from_df(left);
    let rv = TidyView::from_df(right);
    let joined = lv.left_join(&rv, &[("key", "key")]).unwrap();
    assert_eq!(joined.borrow().nrows(), 2);
}

// ── Bench (ignored, same-process headline) ──────────────────────────────

/// Phase 4 cat-aware join vs string-key oracle.
///
/// Run: `cargo test --test test_phase10_tidy --release -- --ignored bench_phase4`.
/// 100k-row left × 100k-row right, 100 unique keys, single-key inner join.
/// Same-process comparison so wall-clock numbers are fair.
#[test]
#[ignore]
fn bench_phase4_categorical_inner_join() {
    use std::time::Instant;

    const N: usize = 100_000;
    const N_LEVELS: usize = 100;
    const ITERS: usize = 5;

    fn build_n_rows(n: usize) -> (DataFrame, DataFrame) {
        let levels: Vec<String> = (0..N_LEVELS).map(|i| format!("L{}", i)).collect();
        let codes: Vec<u32> = (0..n).map(|i| (i % N_LEVELS) as u32).collect();
        let names: Vec<String> = codes.iter().map(|&c| levels[c as usize].clone()).collect();
        let vals: Vec<i64> = (0..n as i64).collect();
        let str_df = DataFrame::from_columns(vec![
            ("key".into(), Column::Str(names)),
            ("v".into(), Column::Int(vals.clone())),
        ])
        .unwrap();
        let cat_df = DataFrame::from_columns(vec![
            ("key".into(), Column::Categorical { levels, codes }),
            ("v".into(), Column::Int(vals)),
        ])
        .unwrap();
        (str_df, cat_df)
    }

    let (l_str, l_cat) = build_n_rows(N);
    let (r_str, r_cat) = build_n_rows(N);
    let lv_str = TidyView::from_df(l_str);
    let rv_str = TidyView::from_df(r_str);
    let lv_cat = TidyView::from_df(l_cat);
    let rv_cat = TidyView::from_df(r_cat);

    // Warm.
    let _ = lv_str.inner_join(&rv_str, &[("key", "key")]).unwrap();
    let _ = lv_cat.inner_join(&rv_cat, &[("key", "key")]).unwrap();

    let mut str_total = std::time::Duration::ZERO;
    let mut cat_total = std::time::Duration::ZERO;
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = lv_str.inner_join(&rv_str, &[("key", "key")]).unwrap();
        str_total += t.elapsed();

        let t = Instant::now();
        let _ = lv_cat.inner_join(&rv_cat, &[("key", "key")]).unwrap();
        cat_total += t.elapsed();
    }
    let str_avg = str_total / ITERS as u32;
    let cat_avg = cat_total / ITERS as u32;
    let speedup = str_avg.as_secs_f64() / cat_avg.as_secs_f64();
    eprintln!(
        "\n[Phase 4 bench] Inner join, {N} × {N} rows, {N_LEVELS} unique keys (avg of {ITERS}):\n\
         \x20\x20String-key path:        {str_avg:?}\n\
         \x20\x20Cat-aware path:         {cat_avg:?}\n\
         \x20\x20Speedup:                {speedup:.2}×\n",
    );
}
