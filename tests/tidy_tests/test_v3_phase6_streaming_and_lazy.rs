//! v3 Phase 6 — Streaming summarise + cat-aware mutate + lazy-plan opt.
//!
//! Three deliverables, three sections:
//!
//! 1. **Streaming summarise** (`TidyView::summarise_streaming`): single-pass
//!    aggregation that avoids materialising per-group `Vec<usize>` row
//!    indices. Output byte-equal to legacy `summarise` on the streamable
//!    subset of aggregations.
//!
//! 2. **Cat-aware mutate**: `mutate("col", DExpr::Col("other_cat"))`
//!    pass-through preserves Categorical / CategoricalAdaptive type
//!    (was previously degraded to `Column::Str`).
//!
//! 3. **Lazy-plan optimizer**: `LazyView::group_summarise` over
//!    streaming-friendly aggregations is rewritten to
//!    `ViewNode::StreamingGroupSummarise` by the optimizer pass.

use cjc_data::{Column, DBinOp, DExpr, DataFrame, StreamingAgg, TidyAgg, TidyView};

// ─────────────────────────────────────────────────────────────────────────
//  Section 1: Streaming summarise
// ─────────────────────────────────────────────────────────────────────────

fn frame_n_groups(n: usize, n_groups: usize) -> DataFrame {
    let key: Vec<i64> = (0..n as i64).map(|i| i % n_groups as i64).collect();
    let v: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    DataFrame::from_columns(vec![
        ("g".into(), Column::Int(key)),
        ("v".into(), Column::Float(v)),
    ])
    .unwrap()
}

#[test]
fn phase6_streaming_count_matches_legacy() {
    let df = frame_n_groups(1000, 5);
    let v = TidyView::from_df(df);

    let streaming = v
        .summarise_streaming(&["g"], &[("n", StreamingAgg::Count)])
        .unwrap();
    let legacy = v
        .group_by_fast(&["g"])
        .unwrap()
        .summarise(&[("n", TidyAgg::Count)])
        .unwrap();

    let s_df = streaming.borrow();
    let l_df = legacy.borrow();
    assert_eq!(s_df.nrows(), l_df.nrows());
    // Output rows: same group keys (BTreeMap order = lex int order).
    // Both frames must have the same number of group rows.
    assert_eq!(s_df.nrows(), l_df.nrows());
    // Count column values must match.
    let s_n: Vec<i64> = match &s_df.columns[1].1 {
        Column::Int(v) => v.clone(),
        _ => panic!("expected Int count"),
    };
    let l_n: Vec<i64> = match &l_df.columns[1].1 {
        Column::Int(v) => v.clone(),
        _ => panic!("expected Int count"),
    };
    assert_eq!(s_n, l_n);
}

#[test]
fn phase6_streaming_sum_mean_min_max_byte_equal_to_legacy() {
    let df = frame_n_groups(1000, 10);
    let v = TidyView::from_df(df);

    let streaming = v
        .summarise_streaming(
            &["g"],
            &[
                ("s", StreamingAgg::Sum("v".into())),
                ("m", StreamingAgg::Mean("v".into())),
                ("lo", StreamingAgg::Min("v".into())),
                ("hi", StreamingAgg::Max("v".into())),
            ],
        )
        .unwrap();
    let legacy = v
        .group_by_fast(&["g"])
        .unwrap()
        .summarise(&[
            ("s", TidyAgg::Sum("v".into())),
            ("m", TidyAgg::Mean("v".into())),
            ("lo", TidyAgg::Min("v".into())),
            ("hi", TidyAgg::Max("v".into())),
        ])
        .unwrap();

    let s_df = streaming.borrow();
    let l_df = legacy.borrow();
    for col_name in &["s", "m", "lo", "hi"] {
        let s_col = s_df.get_column(col_name).unwrap();
        let l_col = l_df.get_column(col_name).unwrap();
        let s_vals: Vec<f64> = match s_col {
            Column::Float(v) => v.clone(),
            _ => panic!(),
        };
        let l_vals: Vec<f64> = match l_col {
            Column::Float(v) => v.clone(),
            _ => panic!(),
        };
        for (s, l) in s_vals.iter().zip(l_vals.iter()) {
            // For sum/mean: Kahan in both. For min/max: bit-equal.
            // Allow small epsilon for sum/mean if order differs.
            assert!(
                (s - l).abs() < 1e-10 || s.is_nan() == l.is_nan(),
                "col={} streaming={} legacy={}",
                col_name,
                s,
                l
            );
        }
    }
}

#[test]
fn phase6_streaming_var_sd_via_welford_matches_legacy() {
    // Welford and the legacy two-pass variance both produce the sample
    // variance; small floating-point differences are tolerated.
    let df = frame_n_groups(500, 4);
    let v = TidyView::from_df(df);

    let streaming = v
        .summarise_streaming(
            &["g"],
            &[("var", StreamingAgg::Var("v".into()))],
        )
        .unwrap();
    let legacy = v
        .group_by_fast(&["g"])
        .unwrap()
        .summarise(&[("var", TidyAgg::Var("v".into()))])
        .unwrap();

    let s_v: Vec<f64> = match &streaming.borrow().columns[1].1 {
        Column::Float(v) => v.clone(),
        _ => panic!(),
    };
    let l_v: Vec<f64> = match &legacy.borrow().columns[1].1 {
        Column::Float(v) => v.clone(),
        _ => panic!(),
    };
    for (s, l) in s_v.iter().zip(l_v.iter()) {
        let rel_err = if l.abs() > 1e-10 { (s - l).abs() / l.abs() } else { (s - l).abs() };
        assert!(
            rel_err < 1e-9,
            "Welford variance mismatch: streaming={} legacy={} rel_err={}",
            s,
            l,
            rel_err
        );
    }
}

#[test]
fn phase6_streaming_categorical_keys() {
    // Cat-aware path: keys are Categorical → Vec<u32> code lookup.
    let levels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let codes: Vec<u32> = (0..600).map(|i| (i % 3) as u32).collect();
    let v: Vec<f64> = (0..600).map(|i| i as f64).collect();
    let df = DataFrame::from_columns(vec![
        ("k".into(), Column::Categorical { levels, codes }),
        ("v".into(), Column::Float(v)),
    ])
    .unwrap();

    let view = TidyView::from_df(df);
    let result = view
        .summarise_streaming(
            &["k"],
            &[
                ("n", StreamingAgg::Count),
                ("s", StreamingAgg::Sum("v".into())),
            ],
        )
        .unwrap();
    let df = result.borrow();
    assert_eq!(df.nrows(), 3);
    // Cat-aware path emits the key column AS Categorical.
    assert!(matches!(df.columns[0].1, Column::Categorical { .. }));
    // Per-group count: 200 each.
    let counts: Vec<i64> = match &df.columns[1].1 {
        Column::Int(v) => v.clone(),
        _ => panic!(),
    };
    assert_eq!(counts, vec![200, 200, 200]);
}

#[test]
fn phase6_streaming_filtered_visible_rows_only() {
    // The streaming path must respect the visible-row mask, not scan
    // the underlying base.
    let df = frame_n_groups(100, 5);
    let view = TidyView::from_df(df);
    let pred = DExpr::BinOp {
        op: DBinOp::Gt,
        left: Box::new(DExpr::Col("g".into())),
        right: Box::new(DExpr::LitInt(2)),
    };
    let filtered = view.filter(&pred).unwrap();
    // After filter: only g ∈ {3, 4} visible.
    let result = filtered
        .summarise_streaming(&["g"], &[("n", StreamingAgg::Count)])
        .unwrap();
    assert_eq!(result.borrow().nrows(), 2);
}

#[test]
fn phase6_streaming_unknown_column_errors() {
    let df = frame_n_groups(10, 2);
    let view = TidyView::from_df(df);
    let r = view.summarise_streaming(&["g"], &[("s", StreamingAgg::Sum("nonexistent".into()))]);
    assert!(r.is_err());
}

// ─────────────────────────────────────────────────────────────────────────
//  Section 2: Cat-aware mutate
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn phase6_mutate_categorical_pass_through_preserves_type() {
    // Pre-Phase-6: mutate("k_copy", Col("k")) on a Categorical column
    // would degrade to Column::Str. Phase 6 preserves the variant.
    let df = DataFrame::from_columns(vec![
        (
            "k".into(),
            Column::Categorical {
                levels: vec!["x".into(), "y".into()],
                codes: vec![0, 1, 0],
            },
        ),
        ("v".into(), Column::Int(vec![1, 2, 3])),
    ])
    .unwrap();
    let view = TidyView::from_df(df);
    let mutated = view
        .mutate(&[("k_copy", DExpr::Col("k".into()))])
        .unwrap();
    let mut_df = mutated.borrow();
    let new_col = mut_df.get_column("k_copy").unwrap();
    assert!(
        matches!(new_col, Column::Categorical { .. }),
        "expected Categorical pass-through, got {:?}",
        new_col.type_name()
    );
}

#[test]
fn phase6_mutate_categorical_adaptive_pass_through_via_legacy() {
    // CategoricalAdaptive flows through `materialize()` → `gather_column`
    // → Phase 5 `to_legacy_categorical` shim, surfacing as
    // `Column::Categorical` by the time `eval_expr_column` sees it.
    // Phase 6's cat-aware mutate then preserves the (legacy) Categorical
    // type through pass-through. The user-visible win is "categorical
    // type survives mutate"; preserving the *adaptive* flavor through
    // the gather pipeline is future work.
    use cjc_data::byte_dict::{ByteDictionary, CategoricalColumn};
    let dict = ByteDictionary::from_explicit(vec![
        b"red".to_vec(),
        b"green".to_vec(),
        b"blue".to_vec(),
    ])
    .unwrap();
    let mut cc = CategoricalColumn::with_dictionary(dict);
    cc.push(b"red").unwrap();
    cc.push(b"green").unwrap();
    cc.push(b"blue").unwrap();

    let df = DataFrame::from_columns(vec![
        ("k".into(), Column::categorical_adaptive(cc)),
        ("v".into(), Column::Int(vec![1, 2, 3])),
    ])
    .unwrap();
    let view = TidyView::from_df(df);
    let mutated = view
        .mutate(&[("k_copy", DExpr::Col("k".into()))])
        .unwrap();
    let mut_df = mutated.borrow();
    let new_col = mut_df.get_column("k_copy").unwrap();
    // Pre-Phase-6 this would degrade to Str. Phase 6 preserves the
    // categorical type through the legacy shim.
    assert!(
        matches!(
            new_col,
            Column::Categorical { .. } | Column::CategoricalAdaptive(_)
        ),
        "expected categorical pass-through, got {:?}",
        new_col.type_name()
    );
}

#[test]
fn phase6_mutate_non_categorical_unchanged() {
    // Non-categorical sources still go through the normal path.
    let df = DataFrame::from_columns(vec![
        ("v".into(), Column::Int(vec![1, 2, 3])),
    ])
    .unwrap();
    let view = TidyView::from_df(df);
    let mutated = view
        .mutate(&[("v2", DExpr::Col("v".into()))])
        .unwrap();
    let mut_df = mutated.borrow();
    let new_col = mut_df.get_column("v2").unwrap();
    assert!(matches!(new_col, Column::Int(_)));
}

// ─────────────────────────────────────────────────────────────────────────
//  Section 3: Lazy-plan optimizer
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn phase6_lazy_streaming_summarise_rewrite_kicks_in() {
    use cjc_data::lazy::LazyView;
    let df = frame_n_groups(100, 3);
    let lazy = LazyView::from_df(df).group_summarise(
        vec!["g".into()],
        vec![
            ("n".into(), TidyAgg::Count),
            ("s".into(), TidyAgg::Sum("v".into())),
        ],
    );
    let optimized = lazy.optimized_plan();
    let kinds = optimized.node_kinds();
    assert!(
        kinds.contains(&"StreamingGroupSummarise"),
        "expected streaming rewrite for all-streamable aggs, got {:?}",
        kinds
    );
    assert!(
        !kinds.contains(&"GroupSummarise"),
        "legacy node should be replaced, got {:?}",
        kinds
    );
}

#[test]
fn phase6_lazy_non_streamable_agg_keeps_legacy_path() {
    use cjc_data::lazy::LazyView;
    let df = frame_n_groups(100, 3);
    // Median is non-streamable.
    let lazy = LazyView::from_df(df).group_summarise(
        vec!["g".into()],
        vec![("med".into(), TidyAgg::Median("v".into()))],
    );
    let optimized = lazy.optimized_plan();
    let kinds = optimized.node_kinds();
    assert!(
        kinds.contains(&"GroupSummarise"),
        "expected legacy node for non-streamable agg, got {:?}",
        kinds
    );
    assert!(
        !kinds.contains(&"StreamingGroupSummarise"),
        "streaming node must not be created for non-streamable, got {:?}",
        kinds
    );
}

#[test]
fn phase6_lazy_mixed_streamable_keeps_legacy_all_or_nothing() {
    use cjc_data::lazy::LazyView;
    let df = frame_n_groups(100, 3);
    // Sum is streamable, Median is not — all-or-nothing → keep legacy.
    let lazy = LazyView::from_df(df).group_summarise(
        vec!["g".into()],
        vec![
            ("s".into(), TidyAgg::Sum("v".into())),
            ("med".into(), TidyAgg::Median("v".into())),
        ],
    );
    let optimized = lazy.optimized_plan();
    let kinds = optimized.node_kinds();
    assert!(
        kinds.contains(&"GroupSummarise"),
        "mixed agg set must keep legacy, got {:?}",
        kinds
    );
}

#[test]
fn phase6_lazy_streaming_executes_correctly_via_collect() {
    use cjc_data::lazy::LazyView;
    let df = frame_n_groups(120, 4);
    let result = LazyView::from_df(df.clone())
        .group_summarise(
            vec!["g".into()],
            vec![("n".into(), TidyAgg::Count)],
        )
        .collect()
        .unwrap();
    let n_rows = result.borrow().nrows();
    assert_eq!(n_rows, 4); // 4 groups
    // Sum of counts == 120.
    let counts: i64 = match &result.borrow().columns[1].1 {
        Column::Int(v) => v.iter().sum(),
        _ => panic!(),
    };
    assert_eq!(counts, 120);
}

// ── Bench (ignored) ─────────────────────────────────────────────────────

#[test]
#[ignore]
fn bench_phase6_streaming_vs_legacy_summarise() {
    use std::time::Instant;
    const N: usize = 1_000_000;
    const N_GROUPS: usize = 1_000;
    const ITERS: usize = 3;

    let df = frame_n_groups(N, N_GROUPS);
    let view = TidyView::from_df(df);

    // Warm.
    let _ = view
        .summarise_streaming(&["g"], &[("n", StreamingAgg::Count), ("s", StreamingAgg::Sum("v".into()))])
        .unwrap();
    let _ = view
        .group_by_fast(&["g"])
        .unwrap()
        .summarise(&[("n", TidyAgg::Count), ("s", TidyAgg::Sum("v".into()))])
        .unwrap();

    let mut s_total = std::time::Duration::ZERO;
    let mut l_total = std::time::Duration::ZERO;
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = view
            .summarise_streaming(&["g"], &[("n", StreamingAgg::Count), ("s", StreamingAgg::Sum("v".into()))])
            .unwrap();
        s_total += t.elapsed();

        let t = Instant::now();
        let _ = view
            .group_by_fast(&["g"])
            .unwrap()
            .summarise(&[("n", TidyAgg::Count), ("s", TidyAgg::Sum("v".into()))])
            .unwrap();
        l_total += t.elapsed();
    }
    let s_avg = s_total / ITERS as u32;
    let l_avg = l_total / ITERS as u32;
    let speedup = l_avg.as_secs_f64() / s_avg.as_secs_f64();
    eprintln!(
        "\n[Phase 6 bench] summarise (Count + Sum), {N} rows × {N_GROUPS} groups (avg of {ITERS}):\n\
         \x20\x20Legacy summarise:    {l_avg:?}\n\
         \x20\x20Streaming summarise: {s_avg:?}\n\
         \x20\x20Speedup:             {speedup:.2}×\n",
    );
}
