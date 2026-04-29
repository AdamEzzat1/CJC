//! Bolero fuzz target for v3 Phase 4 — categorical-aware join key path.
//!
//! Property: for any (left, right) shape pair under categorical keys, the
//! cat-aware fast path produces byte-identical output to the string-key
//! oracle.
//!
//! The "shadow frame" technique is reused from Phase 2 / Phase 4
//! integration tests: build two parallel DataFrames over the same logical
//! values — one as `Column::Str` (forces string path), one as
//! `Column::Categorical` (triggers the cat-aware fast path) — and assert
//! the two output frames render identically.

use cjc_data::{Column, DataFrame, TidyView};
use std::panic;

/// Limit cardinality + length so bolero finds counterexamples fast.
const FUZZ_NLEVELS: usize = 8;
const FUZZ_NROWS_LEFT: usize = 64;
const FUZZ_NROWS_RIGHT: usize = 64;

/// Map a fuzz byte to a level index in `0..FUZZ_NLEVELS`.
fn level_idx(b: u8) -> usize {
    (b as usize) % FUZZ_NLEVELS
}

/// Build paired (string-keyed, cat-keyed) DataFrames from a slice of
/// `(level_idx, value_byte)` pairs. The level set is the same on both
/// frames (so the cat-aware path's remap is identity) — disjoint-level
/// safety is covered by the integration suite's
/// `phase4_inner_join_disjoint_dictionaries` test.
fn build_pair(rows: &[(u8, u8)]) -> (DataFrame, DataFrame) {
    let levels: Vec<String> = (0..FUZZ_NLEVELS).map(|i| format!("L{}", i)).collect();
    let codes: Vec<u32> = rows.iter().map(|(k, _)| level_idx(*k) as u32).collect();
    let names: Vec<String> = rows
        .iter()
        .map(|(k, _)| levels[level_idx(*k)].clone())
        .collect();
    let vals: Vec<i64> = rows.iter().map(|(_, v)| *v as i64).collect();

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

fn split_pairs(input: &[u8]) -> Option<(Vec<(u8, u8)>, Vec<(u8, u8)>)> {
    if input.len() < 4 {
        return None;
    }
    let take_left = (FUZZ_NROWS_LEFT * 2).min(input.len() / 2);
    let take_right = (FUZZ_NROWS_RIGHT * 2).min(input.len() / 2);
    let left_bytes = &input[..take_left];
    let right_bytes = &input[take_left..take_left + take_right];
    if left_bytes.len() < 2 || right_bytes.len() < 2 {
        return None;
    }
    let left_pairs: Vec<(u8, u8)> = left_bytes.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    let right_pairs: Vec<(u8, u8)> = right_bytes.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    Some((left_pairs, right_pairs))
}

fn frame_rows(df: &DataFrame) -> Vec<Vec<String>> {
    let n = df.nrows();
    (0..n)
        .map(|r| df.columns.iter().map(|(_, c)| c.get_display(r)).collect())
        .collect()
}

#[test]
fn fuzz_phase4_inner_join_oracle() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let Some((left_pairs, right_pairs)) = split_pairs(input) else {
                return;
            };
            let (l_str, l_cat) = build_pair(&left_pairs);
            let (r_str, r_cat) = build_pair(&right_pairs);

            let lv_str = TidyView::from_df(l_str);
            let rv_str = TidyView::from_df(r_str);
            let lv_cat = TidyView::from_df(l_cat);
            let rv_cat = TidyView::from_df(r_cat);

            let s = lv_str.inner_join(&rv_str, &[("key", "key")]).unwrap();
            let c = lv_cat.inner_join(&rv_cat, &[("key", "key")]).unwrap();
            assert_eq!(frame_rows(&s.borrow()), frame_rows(&c.borrow()));
        });
    });
}

#[test]
fn fuzz_phase4_left_join_oracle() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let Some((left_pairs, right_pairs)) = split_pairs(input) else {
                return;
            };
            let (l_str, l_cat) = build_pair(&left_pairs);
            let (r_str, r_cat) = build_pair(&right_pairs);

            let lv_str = TidyView::from_df(l_str);
            let rv_str = TidyView::from_df(r_str);
            let lv_cat = TidyView::from_df(l_cat);
            let rv_cat = TidyView::from_df(r_cat);

            let s = lv_str.left_join(&rv_str, &[("key", "key")]).unwrap();
            let c = lv_cat.left_join(&rv_cat, &[("key", "key")]).unwrap();
            assert_eq!(frame_rows(&s.borrow()), frame_rows(&c.borrow()));
        });
    });
}

#[test]
fn fuzz_phase4_semi_anti_join_oracle() {
    bolero::check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = panic::catch_unwind(|| {
            let Some((left_pairs, right_pairs)) = split_pairs(input) else {
                return;
            };
            let (l_str, l_cat) = build_pair(&left_pairs);
            let (r_str, r_cat) = build_pair(&right_pairs);

            let lv_str = TidyView::from_df(l_str);
            let rv_str = TidyView::from_df(r_str);
            let lv_cat = TidyView::from_df(l_cat);
            let rv_cat = TidyView::from_df(r_cat);

            // semi-join
            let s = lv_str.semi_join(&rv_str, &[("key", "key")]).unwrap();
            let c = lv_cat.semi_join(&rv_cat, &[("key", "key")]).unwrap();
            assert_eq!(
                frame_rows(&s.materialize().unwrap()),
                frame_rows(&c.materialize().unwrap())
            );

            // anti-join
            let s = lv_str.anti_join(&rv_str, &[("key", "key")]).unwrap();
            let c = lv_cat.anti_join(&rv_cat, &[("key", "key")]).unwrap();
            assert_eq!(
                frame_rows(&s.materialize().unwrap()),
                frame_rows(&c.materialize().unwrap())
            );
        });
    });
}
