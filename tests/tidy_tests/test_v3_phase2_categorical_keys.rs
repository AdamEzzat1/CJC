//! v3 Phase 2 — categorical-aware key path for `group_by` and `distinct`.
//!
//! These tests pin the bit-identity contract: when every key column is
//! `Column::Categorical`, the cat-aware fast path must produce identical
//! output to the string-key path on the same data.
//!
//! The test strategy is the "shadow column" technique: build two parallel
//! DataFrames over the same logical values — one using `Column::Str`
//! (forces the string path), one using `Column::Categorical` (triggers
//! the fast path) — then run `group_by_fast` / `distinct` against both
//! and assert bit-identity on group order, group keys, group rows, and
//! materialised columns.

use cjc_data::{Column, DBinOp, DExpr, DataFrame, GroupedTidyView, TidyView};

fn col(name: &str) -> DExpr {
    DExpr::Col(name.into())
}

fn lit(v: i64) -> DExpr {
    DExpr::LitInt(v)
}

fn binop(op: DBinOp, l: DExpr, r: DExpr) -> DExpr {
    DExpr::BinOp { op, left: Box::new(l), right: Box::new(r) }
}

/// `v > threshold`
fn pred_v_gt(threshold: i64) -> DExpr {
    binop(DBinOp::Gt, col("v"), lit(threshold))
}

/// Build a DataFrame where the value column `key` is `Column::Str`.
fn df_string_keys(rows: &[(&str, i64)]) -> DataFrame {
    let names: Vec<String> = rows.iter().map(|(k, _)| (*k).to_string()).collect();
    let vals: Vec<i64> = rows.iter().map(|(_, v)| *v).collect();
    DataFrame::from_columns(vec![
        ("key".to_string(), Column::Str(names)),
        ("v".to_string(), Column::Int(vals)),
    ])
    .unwrap()
}

/// Build a DataFrame where the value column `key` is `Column::Categorical`.
/// The level table is the **sorted unique** set of names — this is the
/// canonical Categorical layout produced by `forcats` operations.
fn df_categorical_keys(rows: &[(&str, i64)]) -> DataFrame {
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

/// Same as `df_categorical_keys` but with two categorical key columns
/// (`region` and `product`). Returns the categorical DF and a string
/// sibling DF for parity comparison.
fn df_two_cat_keys(rows: &[(&str, &str, i64)]) -> (DataFrame, DataFrame) {
    use std::collections::BTreeSet;
    let regions_unique: BTreeSet<&str> = rows.iter().map(|(r, _, _)| *r).collect();
    let products_unique: BTreeSet<&str> = rows.iter().map(|(_, p, _)| *p).collect();
    let region_levels: Vec<String> = regions_unique.into_iter().map(String::from).collect();
    let product_levels: Vec<String> = products_unique.into_iter().map(String::from).collect();
    let region_codes: Vec<u32> = rows
        .iter()
        .map(|(r, _, _)| region_levels.iter().position(|s| s == r).unwrap() as u32)
        .collect();
    let product_codes: Vec<u32> = rows
        .iter()
        .map(|(_, p, _)| product_levels.iter().position(|s| s == p).unwrap() as u32)
        .collect();
    let vals: Vec<i64> = rows.iter().map(|(_, _, v)| *v).collect();

    let cat_df = DataFrame::from_columns(vec![
        ("region".into(), Column::Categorical { levels: region_levels, codes: region_codes }),
        ("product".into(), Column::Categorical { levels: product_levels, codes: product_codes }),
        ("v".into(), Column::Int(vals.clone())),
    ])
    .unwrap();

    let str_df = DataFrame::from_columns(vec![
        (
            "region".into(),
            Column::Str(rows.iter().map(|(r, _, _)| (*r).to_string()).collect()),
        ),
        (
            "product".into(),
            Column::Str(rows.iter().map(|(_, p, _)| (*p).to_string()).collect()),
        ),
        ("v".into(), Column::Int(vals)),
    ])
    .unwrap();

    (cat_df, str_df)
}

/// Compare two GroupedTidyView outputs for bit-identity on:
///   - key_names
///   - groups.len()
///   - per-group key_values + row_indices in the same slot order
fn assert_grouped_eq(a: &GroupedTidyView, b: &GroupedTidyView) {
    let ai = a.group_index();
    let bi = b.group_index();
    assert_eq!(ai.key_names, bi.key_names, "key_names mismatch");
    assert_eq!(ai.groups.len(), bi.groups.len(), "group count mismatch");
    for (i, (g_a, g_b)) in ai.groups.iter().zip(bi.groups.iter()).enumerate() {
        assert_eq!(
            g_a.key_values, g_b.key_values,
            "group {} key_values mismatch", i
        );
        assert_eq!(
            g_a.row_indices, g_b.row_indices,
            "group {} row_indices mismatch", i
        );
    }
}

fn assert_views_eq(a: &TidyView, b: &TidyView) {
    let ai: Vec<usize> = a.selection().iter_indices().collect();
    let bi: Vec<usize> = b.selection().iter_indices().collect();
    assert_eq!(ai, bi, "selection iter_indices mismatch");
}

// ─── group_by parity ───────────────────────────────────────────────────

#[test]
fn group_by_single_cat_key_parity_with_string_key() {
    let rows = &[
        ("us", 10), ("eu", 20), ("us", 30), ("ap", 40), ("eu", 50),
        ("us", 60), ("ap", 70), ("us", 80),
    ];
    let cat_df = df_categorical_keys(rows);
    let str_df = df_string_keys(rows);

    let cat_grp = cat_df.tidy().group_by_fast(&["key"]).unwrap();
    let str_grp = str_df.tidy().group_by_fast(&["key"]).unwrap();
    assert_grouped_eq(&cat_grp, &str_grp);
}

#[test]
fn group_by_two_cat_keys_parity_with_string_keys() {
    let rows = &[
        ("us", "x", 1), ("eu", "x", 2), ("us", "y", 3), ("us", "x", 4),
        ("eu", "y", 5), ("ap", "x", 6), ("us", "y", 7), ("eu", "x", 8),
    ];
    let (cat_df, str_df) = df_two_cat_keys(rows);
    let cat_grp = cat_df.tidy().group_by_fast(&["region", "product"]).unwrap();
    let str_grp = str_df.tidy().group_by_fast(&["region", "product"]).unwrap();
    assert_grouped_eq(&cat_grp, &str_grp);
}

#[test]
fn group_by_cat_after_filter_parity() {
    // Cat-aware path should also work after a filter has produced a
    // sparse mask. We pick rows where v > 30.
    let rows = &[
        ("us", 10), ("eu", 20), ("us", 30), ("ap", 40), ("eu", 50),
        ("us", 60), ("ap", 70), ("us", 80),
    ];
    let cat_df = df_categorical_keys(rows);
    let str_df = df_string_keys(rows);

    let cat_view = cat_df.tidy().filter(&pred_v_gt(30)).unwrap();
    let str_view = str_df.tidy().filter(&pred_v_gt(30)).unwrap();

    let cat_grp = cat_view.group_by_fast(&["key"]).unwrap();
    let str_grp = str_view.group_by_fast(&["key"]).unwrap();
    assert_grouped_eq(&cat_grp, &str_grp);
}

#[test]
fn group_by_mixed_cat_and_int_falls_back_to_string_path() {
    // When one key column is categorical and another is int, the
    // cat-aware fast path returns None → fall back to string path.
    // The output must still be correct; we just verify it works.
    let cat_df = DataFrame::from_columns(vec![
        ("region".into(), Column::Categorical {
            levels: vec!["a".into(), "b".into()],
            codes: vec![0, 1, 0, 1, 0],
        }),
        ("year".into(), Column::Int(vec![2020, 2020, 2021, 2021, 2020])),
        ("v".into(), Column::Int(vec![1, 2, 3, 4, 5])),
    ])
    .unwrap();
    let grp = cat_df.tidy().group_by_fast(&["region", "year"]).unwrap();
    let idx = grp.group_index();
    assert_eq!(idx.groups.len(), 4);
    let total_rows: usize = idx.groups.iter().map(|g| g.row_indices.len()).sum();
    assert_eq!(total_rows, 5);
}

#[test]
fn group_by_single_row_cat() {
    let rows = &[("us", 42)];
    let cat_grp = df_categorical_keys(rows).tidy().group_by_fast(&["key"]).unwrap();
    let str_grp = df_string_keys(rows).tidy().group_by_fast(&["key"]).unwrap();
    assert_grouped_eq(&cat_grp, &str_grp);
}

#[test]
fn group_by_empty_view_after_filter_cat() {
    let rows = &[("us", 1), ("eu", 2)];
    let cat_view = df_categorical_keys(rows).tidy().filter(&pred_v_gt(1000)).unwrap();
    let cat_grp = cat_view.group_by_fast(&["key"]).unwrap();
    assert_eq!(cat_grp.group_index().groups.len(), 0);
}

#[test]
fn group_by_all_same_cat_value() {
    let rows = &[("us", 1), ("us", 2), ("us", 3), ("us", 4)];
    let cat_grp = df_categorical_keys(rows).tidy().group_by_fast(&["key"]).unwrap();
    let str_grp = df_string_keys(rows).tidy().group_by_fast(&["key"]).unwrap();
    assert_grouped_eq(&cat_grp, &str_grp);
    assert_eq!(cat_grp.group_index().groups.len(), 1);
    assert_eq!(cat_grp.group_index().groups[0].row_indices, vec![0, 1, 2, 3]);
}

// ─── distinct parity ───────────────────────────────────────────────────

#[test]
fn distinct_single_cat_key_parity() {
    let rows = &[
        ("us", 1), ("eu", 2), ("us", 3), ("ap", 4), ("eu", 5), ("us", 6),
    ];
    let cat_view = df_categorical_keys(rows).tidy().distinct(&["key"]).unwrap();
    let str_view = df_string_keys(rows).tidy().distinct(&["key"]).unwrap();
    assert_views_eq(&cat_view, &str_view);
    let idx: Vec<usize> = cat_view.selection().iter_indices().collect();
    assert_eq!(idx, vec![0, 1, 3], "first-occurrence: us, eu, ap");
}

#[test]
fn distinct_two_cat_keys_parity() {
    let rows = &[
        ("us", "x", 1), ("eu", "x", 2), ("us", "y", 3), ("us", "x", 4),
        ("eu", "y", 5), ("us", "x", 6),
    ];
    let (cat_df, str_df) = df_two_cat_keys(rows);
    let cat_view = cat_df.tidy().distinct(&["region", "product"]).unwrap();
    let str_view = str_df.tidy().distinct(&["region", "product"]).unwrap();
    assert_views_eq(&cat_view, &str_view);
}

#[test]
fn distinct_cat_after_filter_parity() {
    let rows = &[
        ("us", 10), ("eu", 20), ("us", 30), ("ap", 40), ("eu", 50),
        ("us", 60), ("ap", 70), ("us", 80),
    ];
    let cat_view = df_categorical_keys(rows)
        .tidy()
        .filter(&pred_v_gt(40))
        .unwrap()
        .distinct(&["key"])
        .unwrap();
    let str_view = df_string_keys(rows)
        .tidy()
        .filter(&pred_v_gt(40))
        .unwrap()
        .distinct(&["key"])
        .unwrap();
    assert_views_eq(&cat_view, &str_view);
}

// ─── design-validation benchmark ───────────────────────────────────────
//
// This bench is `#[ignore]`d so it doesn't slow the regression gate, but
// it's the headline number for Phase 2. Run with:
//   cargo test --test test_phase10_tidy --release \
//     test_v3_phase2_categorical_keys::bench_cat_vs_string_group_by \
//     -- --ignored --nocapture
//
// Setup: 1M rows, 100 distinct keys (cardinality typical of region /
// product / segment columns). Two parallel DataFrames — one Str, one
// Categorical — with bit-identical visible content. We measure
// group_by_fast wall-time for each. The cat-aware path must:
//   1. Produce identical output (asserted)
//   2. Not be slower than the string path (typical workloads see
//      meaningful speedup; we don't pin a hard ratio because that
//      depends on cache + allocator behaviour).

#[test]
#[ignore = "design-validation bench; run with --ignored --nocapture"]
fn bench_cat_vs_string_group_by() {
    use std::time::Instant;
    let n: usize = 1_000_000;
    let n_unique: usize = 100;

    // Build matched DFs
    let names: Vec<String> = (0..n).map(|i| format!("k_{:03}", i % n_unique)).collect();
    let vals: Vec<i64> = (0..n as i64).collect();

    let str_df = DataFrame::from_columns(vec![
        ("key".into(), Column::Str(names.clone())),
        ("v".into(), Column::Int(vals.clone())),
    ])
    .unwrap();

    let levels: Vec<String> = {
        let mut s: Vec<String> = (0..n_unique).map(|i| format!("k_{:03}", i)).collect();
        s.sort();
        s
    };
    let codes: Vec<u32> = (0..n).map(|i| (i % n_unique) as u32).collect();
    let cat_df = DataFrame::from_columns(vec![
        ("key".into(), Column::Categorical { levels, codes }),
        ("v".into(), Column::Int(vals)),
    ])
    .unwrap();

    // Warm caches
    let _ = cat_df.clone().tidy().group_by_fast(&["key"]).unwrap();
    let _ = str_df.clone().tidy().group_by_fast(&["key"]).unwrap();

    let runs = 5usize;
    let mut cat_total = std::time::Duration::ZERO;
    let mut str_total = std::time::Duration::ZERO;

    for _ in 0..runs {
        let t = Instant::now();
        let cat_grp = cat_df.clone().tidy().group_by_fast(&["key"]).unwrap();
        cat_total += t.elapsed();
        std::hint::black_box(&cat_grp);

        let t = Instant::now();
        let str_grp = str_df.clone().tidy().group_by_fast(&["key"]).unwrap();
        str_total += t.elapsed();
        std::hint::black_box(&str_grp);
    }

    let cat_avg = cat_total / runs as u32;
    let str_avg = str_total / runs as u32;
    let speedup = str_avg.as_secs_f64() / cat_avg.as_secs_f64();

    println!(
        "BENCH cat-aware group_by 1M rows × 100 unique:\n  string:  {:?}\n  cat:     {:?}\n  speedup: {:.2}x",
        str_avg, cat_avg, speedup
    );

    // Parity check on the bench inputs (consumes both DFs)
    let cat_grp = cat_df.tidy().group_by_fast(&["key"]).unwrap();
    let str_grp = str_df.tidy().group_by_fast(&["key"]).unwrap();
    assert_grouped_eq(&cat_grp, &str_grp);
}

#[test]
fn distinct_mixed_cat_and_int_falls_back() {
    // Mixed-type keys must still produce the right answer via the
    // string fallback path.
    let cat_df = DataFrame::from_columns(vec![
        ("region".into(), Column::Categorical {
            levels: vec!["a".into(), "b".into()],
            codes: vec![0, 1, 0, 1, 0],
        }),
        ("year".into(), Column::Int(vec![2020, 2020, 2021, 2021, 2020])),
    ])
    .unwrap();
    let view = cat_df.tidy().distinct(&["region", "year"]).unwrap();
    let idx: Vec<usize> = view.selection().iter_indices().collect();
    assert_eq!(idx.len(), 4);
}
