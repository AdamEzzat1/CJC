//! TidyView Benchmark Suite — Performance, Scaling, Trust, and Auditability
//!
//! Benchmark categories:
//!   A. Filter microbenchmarks (int, float, string predicates; lazy vs materialized)
//!   B. Chained filter composition
//!   C. Group-by + summarise (varying cardinality and aggregation complexity)
//!   D. Arrange / stable sort (int, float, string columns)
//!   E. Join benchmarks (varying right-table size, string vs int keys)
//!   F. Full pipeline benchmarks (realistic multi-step compositions)
//!   G. Scaling benchmarks (row count, column count / width)
//!   H. Determinism verification (bit-identical repeated runs)
//!   I. Memory budget (bitmask allocation proof)
//!   J. Trust / audit trail examples (filter tracing, group inspection,
//!      snapshot semantics, copy-on-write isolation, full pipeline traceability)

use cjc_data::{ArrangeKey, Column, DBinOp, DExpr, DataFrame, TidyAgg};
use std::time::{Duration, Instant};

// ── Constants ───────────────────────────────────────────────────────────────

const N_100K: usize = 100_000;
const N_1M: usize = 1_000_000;
const WARMUP: usize = 3;
const RUNS: usize = 11; // odd for clean median

// ── Dataset Generators ──────────────────────────────────────────────────────

/// Standard 4-column dataset: id(Int), value(Float), category(Str), region(Str)
fn make_df(n: usize) -> DataFrame {
    let ids: Vec<i64> = (0..n as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
    let categories: Vec<String> = (0..n).map(|i| format!("cat_{}", i % 100)).collect();
    let regions: Vec<String> = (0..n)
        .map(|i| match i % 4 {
            0 => "North".into(),
            1 => "South".into(),
            2 => "East".into(),
            _ => "West".into(),
        })
        .collect();
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(ids)),
        ("value".into(), Column::Float(values)),
        ("category".into(), Column::Str(categories)),
        ("region".into(), Column::Str(regions)),
    ])
    .unwrap()
}

/// Wide dataset: id + N float columns
fn make_wide_df(n_rows: usize, n_cols: usize) -> DataFrame {
    let mut cols: Vec<(String, Column)> = Vec::with_capacity(n_cols + 1);
    cols.push(("id".into(), Column::Int((0..n_rows as i64).collect())));
    for c in 0..n_cols {
        let data: Vec<f64> = (0..n_rows).map(|i| (i * (c + 1)) as f64 * 0.01).collect();
        cols.push((format!("col_{}", c), Column::Float(data)));
    }
    DataFrame::from_columns(cols).unwrap()
}

/// High-cardinality string grouping dataset
fn make_high_card_df(n_rows: usize, n_groups: usize) -> DataFrame {
    let ids: Vec<i64> = (0..n_rows as i64).collect();
    let values: Vec<f64> = (0..n_rows).map(|i| i as f64 * 0.1).collect();
    let keys: Vec<String> = (0..n_rows).map(|i| format!("grp_{:06}", i % n_groups)).collect();
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(ids)),
        ("value".into(), Column::Float(values)),
        ("key".into(), Column::Str(keys)),
    ])
    .unwrap()
}

/// Join right-side table with N entries
fn make_join_right(n_categories: usize) -> DataFrame {
    let cats: Vec<String> = (0..n_categories).map(|i| format!("cat_{}", i)).collect();
    let scores: Vec<f64> = (0..n_categories).map(|i| i as f64 * 10.0).collect();
    DataFrame::from_columns(vec![
        ("category".into(), Column::Str(cats)),
        ("score".into(), Column::Float(scores)),
    ])
    .unwrap()
}

// ── Expression Helpers ──────────────────────────────────────────────────────

fn col(name: &str) -> DExpr { DExpr::Col(name.into()) }

fn gt_int(c: &str, v: i64) -> DExpr {
    DExpr::BinOp { op: DBinOp::Gt, left: Box::new(col(c)), right: Box::new(DExpr::LitInt(v)) }
}

fn gt_float(c: &str, v: f64) -> DExpr {
    DExpr::BinOp { op: DBinOp::Gt, left: Box::new(col(c)), right: Box::new(DExpr::LitFloat(v)) }
}

fn eq_str(c: &str, v: &str) -> DExpr {
    DExpr::BinOp { op: DBinOp::Eq, left: Box::new(col(c)), right: Box::new(DExpr::LitStr(v.into())) }
}

// ── Bench Harness ───────────────────────────────────────────────────────────

fn median_of(samples: &mut Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn bench<F: FnMut()>(mut f: F) -> Duration {
    for _ in 0..WARMUP { f(); }
    let mut s: Vec<Duration> = (0..RUNS).map(|_| { let t = Instant::now(); f(); t.elapsed() }).collect();
    median_of(&mut s)
}

fn us(d: Duration) -> u64 { d.as_micros() as u64 }
fn ms(d: Duration) -> f64 { d.as_micros() as f64 / 1000.0 }

// ════════════════════════════════════════════════════════════════════════════
// A. FILTER MICROBENCHMARKS
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_a1_filter_int_bitmask_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let pred = gt_int("id", N_100K as i64 / 2);

    let lazy = bench(|| { let _ = view.filter(&pred).unwrap(); });
    let eager = bench(|| { let _ = view.filter(&pred).unwrap().materialize().unwrap(); });
    let ratio = eager.as_nanos() as f64 / lazy.as_nanos().max(1) as f64;

    println!("RESULT:A1:filter_int_100k:lazy_us:{}:eager_us:{}:ratio:{:.1}x", us(lazy), us(eager), ratio);
}

#[test]
fn bench_a2_filter_float_bitmask_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let pred = gt_float("value", 50.0);

    let lazy = bench(|| { let _ = view.filter(&pred).unwrap(); });
    let eager = bench(|| { let _ = view.filter(&pred).unwrap().materialize().unwrap(); });
    let ratio = eager.as_nanos() as f64 / lazy.as_nanos().max(1) as f64;

    println!("RESULT:A2:filter_float_100k:lazy_us:{}:eager_us:{}:ratio:{:.1}x", us(lazy), us(eager), ratio);
}

#[test]
fn bench_a3_filter_string_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let pred = eq_str("region", "West");

    let lazy = bench(|| { let _ = view.filter(&pred).unwrap(); });
    let eager = bench(|| { let _ = view.filter(&pred).unwrap().materialize().unwrap(); });

    println!("RESULT:A3:filter_string_100k:lazy_us:{}:eager_us:{}", us(lazy), us(eager));
}

#[test]
fn bench_a4_filter_int_bitmask_1m() {
    let df = make_df(N_1M);
    let view = df.tidy();
    let pred = gt_int("id", N_1M as i64 / 2);

    let lazy = bench(|| { let _ = view.filter(&pred).unwrap(); });
    let eager = bench(|| { let _ = view.filter(&pred).unwrap().materialize().unwrap(); });
    let ratio = eager.as_nanos() as f64 / lazy.as_nanos().max(1) as f64;

    println!("RESULT:A4:filter_int_1m:lazy_us:{}:eager_us:{}:ratio:{:.1}x", us(lazy), us(eager), ratio);
}

// ════════════════════════════════════════════════════════════════════════════
// B. CHAINED FILTER COMPOSITION
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_b1_chain_int_int_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p1 = gt_int("id", N_100K as i64 / 4);
    let p2 = gt_int("id", N_100K as i64 / 2);

    let single = bench(|| { let _ = view.filter(&p1).unwrap(); });
    let chained = bench(|| {
        let v = view.filter(&p1).unwrap();
        let _ = v.filter(&p2).unwrap();
    });

    println!("RESULT:B1:chain_int_int_100k:single_us:{}:chained_us:{}", us(single), us(chained));
}

#[test]
fn bench_b2_chain_int_string_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p1 = gt_int("id", N_100K as i64 / 4);
    let p2 = eq_str("region", "West");

    let chained = bench(|| {
        let v = view.filter(&p1).unwrap();
        let _ = v.filter(&p2).unwrap();
    });

    println!("RESULT:B2:chain_int_str_100k:us:{}", us(chained));
}

#[test]
fn bench_b3_chain_string_string_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p1 = eq_str("region", "West");
    let p2 = eq_str("category", "cat_0");

    let chained = bench(|| {
        let v = view.filter(&p1).unwrap();
        let _ = v.filter(&p2).unwrap();
    });

    println!("RESULT:B3:chain_str_str_100k:us:{}", us(chained));
}

// ════════════════════════════════════════════════════════════════════════════
// C. GROUP-BY + SUMMARISE (varying cardinality)
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_c1_group_4_count_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| {
        let g = view.group_by(&["region"]).unwrap();
        let _ = g.summarise(&[("n", TidyAgg::Count)]).unwrap();
    });
    println!("RESULT:C1:group_4_count_100k:us:{}", us(p50));
}

#[test]
fn bench_c2_group_4_mixed_agg_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| {
        let g = view.group_by(&["region"]).unwrap();
        let _ = g.summarise(&[
            ("n", TidyAgg::Count),
            ("total", TidyAgg::Sum("value".into())),
            ("avg", TidyAgg::Mean("value".into())),
            ("sd", TidyAgg::Sd("value".into())),
            ("med", TidyAgg::Median("value".into())),
        ]).unwrap();
    });
    println!("RESULT:C2:group_4_mixed_100k:us:{}", us(p50));
}

#[test]
fn bench_c3_group_100_sum_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| {
        let g = view.group_by(&["category"]).unwrap();
        let _ = g.summarise(&[
            ("n", TidyAgg::Count),
            ("total", TidyAgg::Sum("value".into())),
        ]).unwrap();
    });
    println!("RESULT:C3:group_100_sum_100k:us:{}", us(p50));
}

#[test]
fn bench_c4_group_10k_count_100k() {
    let df = make_high_card_df(N_100K, 10_000);
    let view = df.tidy();
    let p50 = bench(|| {
        let g = view.group_by(&["key"]).unwrap();
        let _ = g.summarise(&[("n", TidyAgg::Count)]).unwrap();
    });
    println!("RESULT:C4:group_10k_count_100k:us:{}", us(p50));
}

#[test]
fn bench_c5_group_10k_sum_100k() {
    let df = make_high_card_df(N_100K, 10_000);
    let view = df.tidy();
    let p50 = bench(|| {
        let g = view.group_by(&["key"]).unwrap();
        let _ = g.summarise(&[
            ("n", TidyAgg::Count),
            ("total", TidyAgg::Sum("value".into())),
            ("avg", TidyAgg::Mean("value".into())),
        ]).unwrap();
    });
    println!("RESULT:C5:group_10k_sum_100k:us:{}", us(p50));
}

// ════════════════════════════════════════════════════════════════════════════
// D. ARRANGE / STABLE SORT
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_d1_sort_float_desc_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| { let _ = view.arrange(&[ArrangeKey::desc("value")]).unwrap(); });
    println!("RESULT:D1:sort_float_desc_100k:us:{}", us(p50));
}

#[test]
fn bench_d2_sort_int_asc_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| { let _ = view.arrange(&[ArrangeKey::asc("id")]).unwrap(); });
    println!("RESULT:D2:sort_int_asc_100k:us:{}", us(p50));
}

#[test]
fn bench_d3_sort_string_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| { let _ = view.arrange(&[ArrangeKey::asc("category")]).unwrap(); });
    println!("RESULT:D3:sort_string_100k:us:{}", us(p50));
}

#[test]
fn bench_d4_sort_multikey_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| {
        let _ = view.arrange(&[ArrangeKey::asc("region"), ArrangeKey::desc("value")]).unwrap();
    });
    println!("RESULT:D4:sort_multikey_100k:us:{}", us(p50));
}

// ════════════════════════════════════════════════════════════════════════════
// E. JOIN BENCHMARKS
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_e1_inner_join_100k_x_100() {
    let left = make_df(N_100K);
    let right = make_join_right(100);
    let lv = left.tidy();
    let rv = right.tidy();
    let p50 = bench(|| { let _ = lv.inner_join(&rv, &[("category", "category")]).unwrap(); });
    println!("RESULT:E1:inner_join_100kx100:us:{}", us(p50));
}

#[test]
fn bench_e2_inner_join_100k_x_1k() {
    // 1000 categories — 100 match, 900 don't
    let left = make_df(N_100K);
    let right = make_join_right(1000);
    let lv = left.tidy();
    let rv = right.tidy();
    let p50 = bench(|| { let _ = lv.inner_join(&rv, &[("category", "category")]).unwrap(); });
    println!("RESULT:E2:inner_join_100kx1k:us:{}", us(p50));
}

#[test]
fn bench_e3_left_join_100k_x_100() {
    let left = make_df(N_100K);
    let right = make_join_right(100);
    let lv = left.tidy();
    let rv = right.tidy();
    let p50 = bench(|| { let _ = lv.left_join(&rv, &[("category", "category")]).unwrap(); });
    println!("RESULT:E3:left_join_100kx100:us:{}", us(p50));
}

// ════════════════════════════════════════════════════════════════════════════
// F. FULL PIPELINE BENCHMARKS
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_f1_filter_group_summarise_arrange_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| {
        let filtered = view.filter(&gt_int("id", N_100K as i64 / 4)).unwrap();
        let grouped = filtered.group_by(&["region"]).unwrap();
        let summary = grouped.summarise(&[
            ("n", TidyAgg::Count),
            ("total", TidyAgg::Sum("value".into())),
            ("avg", TidyAgg::Mean("value".into())),
            ("max_val", TidyAgg::Max("value".into())),
        ]).unwrap();
        let _ = summary.view().arrange(&[ArrangeKey::desc("total")]).unwrap();
    });
    println!("RESULT:F1:filter_group_summarise_arrange_100k:us:{}", us(p50));
}

#[test]
fn bench_f2_filter_mutate_group_summarise_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let p50 = bench(|| {
        let filtered = view.filter(&gt_int("id", N_100K as i64 / 2)).unwrap();
        let mutated = filtered.mutate(&[(
            "scaled",
            DExpr::BinOp {
                op: DBinOp::Mul,
                left: Box::new(col("value")),
                right: Box::new(DExpr::LitFloat(100.0)),
            },
        )]).unwrap();
        let grouped = mutated.view().group_by(&["region"]).unwrap();
        let _ = grouped.summarise(&[
            ("n", TidyAgg::Count),
            ("total_scaled", TidyAgg::Sum("scaled".into())),
        ]).unwrap();
    });
    println!("RESULT:F2:filter_mutate_group_summarise_100k:us:{}", us(p50));
}

#[test]
fn bench_f3_join_filter_group_summarise_100k() {
    let left = make_df(N_100K);
    let right = make_join_right(100);
    let lv = left.tidy();
    let rv = right.tidy();
    let p50 = bench(|| {
        let joined = lv.inner_join(&rv, &[("category", "category")]).unwrap();
        let filtered = joined.view().filter(&gt_float("score", 500.0)).unwrap();
        let grouped = filtered.group_by(&["region"]).unwrap();
        let _ = grouped.summarise(&[
            ("n", TidyAgg::Count),
            ("avg_score", TidyAgg::Mean("score".into())),
        ]).unwrap();
    });
    println!("RESULT:F3:join_filter_group_summarise_100k:us:{}", us(p50));
}

// ════════════════════════════════════════════════════════════════════════════
// G. SCALING BENCHMARKS
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_g1_filter_row_scaling() {
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let df = make_df(n);
        let view = df.tidy();
        let pred = gt_int("id", n as i64 / 2);
        let p50 = bench(|| { let _ = view.filter(&pred).unwrap(); });
        println!("RESULT:G1:filter_row_scaling:n:{}:us:{}", n, us(p50));
    }
}

#[test]
fn bench_g2_group_row_scaling() {
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let df = make_df(n);
        let view = df.tidy();
        let p50 = bench(|| {
            let g = view.group_by(&["region"]).unwrap();
            let _ = g.summarise(&[
                ("n", TidyAgg::Count),
                ("total", TidyAgg::Sum("value".into())),
            ]).unwrap();
        });
        println!("RESULT:G2:group_row_scaling:n:{}:us:{}", n, us(p50));
    }
}

#[test]
fn bench_g3_column_width_scaling() {
    let n_rows = 100_000;
    for &n_cols in &[2usize, 10, 50, 100] {
        let df = make_wide_df(n_rows, n_cols);
        let view = df.tidy();
        let pred = gt_int("id", n_rows as i64 / 2);

        let filter_p50 = bench(|| { let _ = view.filter(&pred).unwrap(); });
        let mat_p50 = bench(|| { let _ = view.filter(&pred).unwrap().materialize().unwrap(); });

        println!(
            "RESULT:G3:width_scaling:cols:{}:filter_us:{}:materialize_us:{}",
            n_cols + 1, us(filter_p50), us(mat_p50)
        );
    }
}

#[test]
fn bench_g4_group_cardinality_scaling() {
    let n = 100_000;
    for &g in &[4usize, 100, 1_000, 10_000, 50_000] {
        let df = make_high_card_df(n, g);
        let view = df.tidy();
        let p50 = bench(|| {
            let grouped = view.group_by(&["key"]).unwrap();
            let _ = grouped.summarise(&[
                ("n", TidyAgg::Count),
                ("total", TidyAgg::Sum("value".into())),
            ]).unwrap();
        });
        println!("RESULT:G4:group_card_scaling:groups:{}:us:{}", g, us(p50));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// H. DETERMINISM VERIFICATION
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_h1_determinism_pipeline_100k() {
    let df = make_df(N_100K);
    let view = df.tidy();
    let mut all_results: Vec<Vec<f64>> = Vec::new();

    for _ in 0..10 {
        let filtered = view.filter(&gt_int("id", N_100K as i64 / 4)).unwrap();
        let grouped = filtered.group_by(&["region"]).unwrap();
        let summary = grouped.summarise(&[
            ("total", TidyAgg::Sum("value".into())),
            ("avg", TidyAgg::Mean("value".into())),
            ("sd", TidyAgg::Sd("value".into())),
            ("med", TidyAgg::Median("value".into())),
        ]).unwrap();
        let out = summary.borrow();
        // Collect all float columns into one flat vec for comparison
        let mut vals = Vec::new();
        for (_, c) in &out.columns {
            if let Column::Float(v) = c { vals.extend(v); }
        }
        all_results.push(vals);
    }

    for i in 1..all_results.len() {
        assert_eq!(all_results[0], all_results[i],
            "Run 0 vs run {}: determinism violated", i);
    }
    println!("RESULT:H1:determinism_100k:PASS:10_runs_bit_identical");
}

#[test]
fn bench_h2_determinism_sort_stability() {
    // Same values for sort key → stable sort must preserve original order
    let df = DataFrame::from_columns(vec![
        ("key".into(), Column::Int(vec![3, 1, 2, 1, 3, 2])),
        ("order".into(), Column::Int(vec![0, 1, 2, 3, 4, 5])),
    ]).unwrap();
    let view = df.tidy();

    let mut results: Vec<Vec<i64>> = Vec::new();
    for _ in 0..5 {
        let sorted = view.arrange(&[ArrangeKey::asc("key")]).unwrap();
        let mat = sorted.materialize().unwrap();
        if let Column::Int(v) = &mat.columns[1].1 {
            results.push(v.clone());
        }
    }

    // All runs must be identical AND must preserve insertion order within equal keys
    assert_eq!(results[0], vec![1, 3, 2, 5, 0, 4]); // key=1 first (rows 1,3), key=2 (2,5), key=3 (0,4)
    for i in 1..results.len() {
        assert_eq!(results[0], results[i]);
    }
    println!("RESULT:H2:sort_stability:PASS:stable_order_preserved");
}

#[test]
fn bench_h3_determinism_group_order() {
    let df = DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec![
            "Sales".into(), "Eng".into(), "HR".into(),
            "Eng".into(), "Sales".into(), "HR".into(),
        ])),
        ("val".into(), Column::Int(vec![10, 20, 30, 40, 50, 60])),
    ]).unwrap();
    let view = df.tidy();

    let mut group_orders: Vec<Vec<String>> = Vec::new();
    for _ in 0..5 {
        let g = view.group_by(&["dept"]).unwrap();
        let s = g.summarise(&[("total", TidyAgg::Sum("val".into()))]).unwrap();
        let out = s.borrow();
        if let Column::Str(v) = &out.columns[0].1 {
            group_orders.push(v.clone());
        }
    }

    // First-occurrence: Sales (row 0), Eng (row 1), HR (row 2)
    assert_eq!(group_orders[0], vec!["Sales", "Eng", "HR"]);
    for i in 1..group_orders.len() {
        assert_eq!(group_orders[0], group_orders[i]);
    }
    println!("RESULT:H3:group_order:PASS:first_occurrence_deterministic");
}

// ════════════════════════════════════════════════════════════════════════════
// I. MEMORY BUDGET
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_i1_filter_allocation_budget_1m() {
    let df = make_df(N_1M);
    let view = df.tidy();
    let filtered = view.filter(&gt_int("id", 0)).unwrap();

    let mask_words = filtered.mask().nwords();
    let expected_words = (N_1M + 63) / 64;
    let mask_bytes = mask_words * 8;
    // Data = id(8 bytes) + value(8 bytes) per row minimum
    let data_bytes = N_1M * 16;

    assert_eq!(mask_words, expected_words);
    let ratio = data_bytes as f64 / mask_bytes as f64;
    println!("RESULT:I1:alloc_budget_1m:mask_kb:{}:data_kb:{}:ratio:{:.0}x",
        mask_bytes / 1024, data_bytes / 1024, ratio);
}

#[test]
fn bench_i2_chained_filter_no_data_copy() {
    let df = make_df(N_100K);
    let view = df.tidy();

    let f1 = view.filter(&gt_int("id", N_100K as i64 / 4)).unwrap();
    let f2 = f1.filter(&gt_int("id", N_100K as i64 / 2)).unwrap();
    let f3 = f2.filter(&gt_int("id", 3 * N_100K as i64 / 4)).unwrap();

    // All three views share the same projection
    assert_eq!(view.proj().len(), f1.proj().len());
    assert_eq!(view.proj().len(), f2.proj().len());
    assert_eq!(view.proj().len(), f3.proj().len());

    // Masks are independent but all O(N/64) words
    let expected = (N_100K + 63) / 64;
    assert_eq!(f1.mask().nwords(), expected);
    assert_eq!(f2.mask().nwords(), expected);
    assert_eq!(f3.mask().nwords(), expected);

    // Row counts shrink progressively
    assert!(f1.nrows() > f2.nrows());
    assert!(f2.nrows() > f3.nrows());

    println!("RESULT:I2:chained_no_copy:views:3:mask_words_each:{}:rows:{}/{}/{}",
        expected, f1.nrows(), f2.nrows(), f3.nrows());
}

// ════════════════════════════════════════════════════════════════════════════
// J. TRUST / AUDIT TRAIL EXAMPLES
// ════════════════════════════════════════════════════════════════════════════

/// Filter row traceability: see exactly which rows survived each step.
#[test]
fn audit_j1_filter_traceability() {
    let df = DataFrame::from_columns(vec![
        ("name".into(), Column::Str(vec![
            "Alice".into(), "Bob".into(), "Carol".into(), "Dave".into(),
            "Eve".into(), "Frank".into(), "Grace".into(), "Heidi".into(),
            "Ivan".into(), "Judy".into(),
        ])),
        ("dept".into(), Column::Str(vec![
            "Eng".into(), "Sales".into(), "Eng".into(), "Sales".into(),
            "Eng".into(), "Sales".into(), "Eng".into(), "Sales".into(),
            "Eng".into(), "Sales".into(),
        ])),
        ("salary".into(), Column::Int(vec![120, 80, 95, 110, 130, 75, 100, 105, 115, 90])),
    ]).unwrap();

    let view = df.tidy();

    // Step 1: salary > 100
    let step1 = view.filter(&gt_int("salary", 100)).unwrap();
    let vis1: Vec<usize> = step1.mask().iter_set().collect();
    assert_eq!(vis1, vec![0, 3, 4, 7, 8]);

    // Step 2: dept == "Eng"
    let step2 = step1.filter(&eq_str("dept", "Eng")).unwrap();
    let vis2: Vec<usize> = step2.mask().iter_set().collect();
    assert_eq!(vis2, vec![0, 4, 8]);

    // Original unchanged
    assert_eq!(view.nrows(), 10);

    // Row-level trace
    let mat = view.materialize().unwrap();
    if let Column::Str(names) = &mat.columns[0].1 {
        assert_eq!(names[0], "Alice");
        assert_eq!(names[4], "Eve");
        assert_eq!(names[8], "Ivan");
    }

    println!("RESULT:J1:filter_trace:original:10:step1:{}:step2:{}:indices:{:?}",
        step1.nrows(), step2.nrows(), vis2);
}

/// Group membership inspection: see exactly which rows are in each group.
#[test]
fn audit_j2_group_membership() {
    let df = DataFrame::from_columns(vec![
        ("dept".into(), Column::Str(vec![
            "Eng".into(), "Sales".into(), "Eng".into(),
            "HR".into(), "Sales".into(), "HR".into(),
        ])),
        ("salary".into(), Column::Int(vec![120, 80, 95, 70, 110, 85])),
    ]).unwrap();

    let view = df.tidy();
    let grouped = view.group_by(&["dept"]).unwrap();
    let index = grouped.group_index();

    // First-occurrence order
    assert_eq!(index.groups[0].key_values, vec!["Eng"]);
    assert_eq!(index.groups[0].row_indices, vec![0, 2]);
    assert_eq!(index.groups[1].key_values, vec!["Sales"]);
    assert_eq!(index.groups[1].row_indices, vec![1, 4]);
    assert_eq!(index.groups[2].key_values, vec!["HR"]);
    assert_eq!(index.groups[2].row_indices, vec![3, 5]);

    let summary = grouped.summarise(&[
        ("total", TidyAgg::Sum("salary".into())),
    ]).unwrap();
    let out = summary.borrow();
    if let Column::Float(totals) = &out.columns[1].1 {
        assert_eq!(totals, &vec![215.0, 190.0, 155.0]);
    }

    println!("RESULT:J2:group_membership:groups:3:order:Eng,Sales,HR:totals:215,190,155");
}

/// Snapshot semantics: mutate can't read columns it creates in the same call.
#[test]
fn audit_j3_snapshot_semantics() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![10, 20, 30])),
    ]).unwrap();

    let view = df.tidy();
    let result = view.mutate(&[
        ("x_plus", DExpr::BinOp {
            op: DBinOp::Add, left: Box::new(col("x")), right: Box::new(DExpr::LitInt(1)),
        }),
        ("bad_ref", DExpr::BinOp {
            op: DBinOp::Mul, left: Box::new(col("x_plus")), right: Box::new(DExpr::LitInt(2)),
        }),
    ]);

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("not found"), "Expected ColumnNotFound, got: {}", err_msg);
    println!("RESULT:J3:snapshot_semantics:PASS:blocked_self_reference:error:{}", err_msg);
}

/// Copy-on-write isolation: cloned frames are independent after mutation.
#[test]
fn audit_j4_copy_on_write() {
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(vec![10, 20, 30])),
    ]).unwrap();

    let view = df.tidy();
    let mut frame_a = view.mutate(&[(
        "doubled", DExpr::BinOp {
            op: DBinOp::Mul, left: Box::new(col("x")), right: Box::new(DExpr::LitInt(2)),
        },
    )]).unwrap();

    let frame_b = frame_a.clone(); // refcount = 2

    // Mutate frame_a — triggers deep clone
    frame_a.mutate(&[(
        "tripled", DExpr::BinOp {
            op: DBinOp::Mul, left: Box::new(col("x")), right: Box::new(DExpr::LitInt(3)),
        },
    )]).unwrap();

    assert_eq!(frame_a.borrow().ncols(), 3); // x, doubled, tripled
    assert_eq!(frame_b.borrow().ncols(), 2); // x, doubled — unaffected

    println!("RESULT:J4:copy_on_write:frame_a_cols:3:frame_b_cols:2:ISOLATED");
}

/// Full pipeline traceability: trace data through filter → group → summarise.
#[test]
fn audit_j5_full_pipeline_trace() {
    let df = DataFrame::from_columns(vec![
        ("product".into(), Column::Str(vec![
            "Widget".into(), "Gadget".into(), "Widget".into(), "Gadget".into(),
            "Widget".into(), "Gizmo".into(), "Gadget".into(), "Gizmo".into(),
        ])),
        ("region".into(), Column::Str(vec![
            "West".into(), "West".into(), "East".into(), "East".into(),
            "West".into(), "East".into(), "West".into(), "East".into(),
        ])),
        ("revenue".into(), Column::Float(vec![150.0, 200.0, 175.0, 225.0, 300.0, 50.0, 180.0, 75.0])),
    ]).unwrap();

    let view = df.tidy();

    // Filter to West
    let west = view.filter(&eq_str("region", "West")).unwrap();
    let west_rows: Vec<usize> = west.mask().iter_set().collect();
    assert_eq!(west_rows, vec![0, 1, 4, 6]);

    // Group by product
    let grouped = west.group_by(&["product"]).unwrap();
    let idx = grouped.group_index();
    assert_eq!(idx.groups[0].key_values, vec!["Widget"]);
    assert_eq!(idx.groups[0].row_indices, vec![0, 4]);
    assert_eq!(idx.groups[1].key_values, vec!["Gadget"]);
    assert_eq!(idx.groups[1].row_indices, vec![1, 6]);

    // Summarise
    let summary = grouped.summarise(&[
        ("total_rev", TidyAgg::Sum("revenue".into())),
    ]).unwrap();
    let out = summary.borrow();
    if let Column::Float(totals) = &out.columns[1].1 {
        assert_eq!(totals[0], 450.0); // Widget: 150 + 300
        assert_eq!(totals[1], 380.0); // Gadget: 200 + 180
    }

    // Original unchanged
    assert_eq!(view.nrows(), 8);

    println!("RESULT:J5:full_trace:west_rows:{:?}:widget_rev:450:gadget_rev:380:original_intact:8",
        west_rows);
}

/// Deterministic sampling: slice_sample with same seed = same rows every time.
#[test]
fn audit_j6_deterministic_sampling() {
    let df = make_df(N_100K);
    let view = df.tidy();

    let mut sample_rows: Vec<Vec<usize>> = Vec::new();
    for _ in 0..5 {
        let sampled = view.slice_sample(100, 42);
        let rows: Vec<usize> = sampled.mask().iter_set().collect();
        sample_rows.push(rows);
    }

    for i in 1..sample_rows.len() {
        assert_eq!(sample_rows[0], sample_rows[i],
            "Sample run 0 vs run {}: different rows selected", i);
    }

    println!("RESULT:J6:deterministic_sample:PASS:5_runs_identical:n_sampled:{}", sample_rows[0].len());
}

/// Projection audit: select only changes which columns are visible, not the data.
#[test]
fn audit_j7_projection_inspection() {
    let df = DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![1, 2, 3])),
        ("b".into(), Column::Int(vec![4, 5, 6])),
        ("c".into(), Column::Int(vec![7, 8, 9])),
        ("d".into(), Column::Int(vec![10, 11, 12])),
    ]).unwrap();

    let view = df.tidy();
    let selected = view.select(&["b", "d"]).unwrap();

    assert_eq!(view.ncols(), 4);
    assert_eq!(selected.ncols(), 2);
    assert_eq!(selected.proj().indices(), &[1, 3]);
    assert_eq!(view.ncols(), 4); // unchanged

    println!("RESULT:J7:projection:original_cols:4:selected_cols:2:proj_indices:[1,3]");
}
