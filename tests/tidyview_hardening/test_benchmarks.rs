//! TidyView performance benchmarks on large data.
//! Tests correctness AND measures wall-clock time for key operations.

use cjc_data::{Column, DataFrame, DExpr, TidyAgg, ArrangeKey};
use cjc_data::lazy::LazyView;
use std::time::Instant;

/// Build a large DataFrame with N rows, 3 groups, and numeric columns.
fn make_large_df(n: usize) -> DataFrame {
    let mut group_col = Vec::with_capacity(n);
    let mut val_col = Vec::with_capacity(n);
    let mut id_col = Vec::with_capacity(n);
    let groups = ["alpha", "beta", "gamma"];
    for i in 0..n {
        group_col.push(groups[i % 3].to_string());
        val_col.push(i as f64 * 1.1);
        id_col.push(i as i64);
    }
    DataFrame::from_columns(vec![
        ("group".into(), Column::Str(group_col)),
        ("value".into(), Column::Float(val_col)),
        ("id".into(), Column::Int(id_col)),
    ])
    .unwrap()
}

/// Build a large DataFrame with many groups for stress testing.
fn make_many_groups_df(n: usize, n_groups: usize) -> DataFrame {
    let mut group_col = Vec::with_capacity(n);
    let mut val_col = Vec::with_capacity(n);
    for i in 0..n {
        group_col.push(format!("g{}", i % n_groups));
        val_col.push(i as f64 * 0.7);
    }
    DataFrame::from_columns(vec![
        ("group".into(), Column::Str(group_col)),
        ("value".into(), Column::Float(val_col)),
    ])
    .unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// Filter benchmarks
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_filter_100k_rows() {
    let df = make_large_df(100_000);
    let view = df.tidy();

    let start = Instant::now();
    // Filter where value > 50000 (roughly half the rows)
    let filtered = view.filter(&DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(DExpr::Col("value".into())),
        right: Box::new(DExpr::LitFloat(55_000.0)),
    }).unwrap();
    let elapsed_filter = start.elapsed();

    // Materialize to check correctness
    let start2 = Instant::now();
    let mat = filtered.materialize().unwrap();
    let elapsed_materialize = start2.elapsed();

    let nrows = mat.nrows();
    assert!(nrows > 0, "filter should produce rows");
    assert!(nrows < 100_000, "filter should remove rows");

    eprintln!(
        "[bench_filter_100k] filter: {:?}, materialize: {:?}, rows: {} -> {}",
        elapsed_filter, elapsed_materialize, 100_000, nrows
    );
    // Should complete in under 1 second
    assert!(elapsed_filter.as_millis() + elapsed_materialize.as_millis() < 5000);
}

#[test]
fn bench_filter_1m_rows() {
    let df = make_large_df(1_000_000);
    let view = df.tidy();

    let start = Instant::now();
    let filtered = view.filter(&DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(DExpr::Col("value".into())),
        right: Box::new(DExpr::LitFloat(550_000.0)),
    }).unwrap();
    let mat = filtered.materialize().unwrap();
    let elapsed = start.elapsed();

    let nrows = mat.nrows();
    assert!(nrows > 0);
    assert!(nrows < 1_000_000);

    eprintln!(
        "[bench_filter_1m] total: {:?}, rows: {} -> {}",
        elapsed, 1_000_000, nrows
    );
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

// ═══════════════════════════════════════════════════════════════════════════
// Select benchmarks
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_select_100k_rows() {
    let df = make_large_df(100_000);
    let view = df.tidy();

    let start = Instant::now();
    let selected = view.select(&["group", "value"]).unwrap();
    let elapsed_select = start.elapsed();

    let start2 = Instant::now();
    let mat = selected.materialize().unwrap();
    let elapsed_materialize = start2.elapsed();

    assert_eq!(mat.ncols(), 2);
    assert_eq!(mat.nrows(), 100_000);

    eprintln!(
        "[bench_select_100k] select: {:?}, materialize: {:?}",
        elapsed_select, elapsed_materialize
    );
    assert!(elapsed_select.as_millis() + elapsed_materialize.as_millis() < 5000);
}

// ═══════════════════════════════════════════════════════════════════════════
// Filter + Select chain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_filter_select_chain_100k() {
    let df = make_large_df(100_000);
    let view = df.tidy();

    let start = Instant::now();
    let result = view
        .filter(&DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(DExpr::Col("value".into())),
            right: Box::new(DExpr::LitFloat(50_000.0)),
        })
        .unwrap()
        .select(&["group", "value"])
        .unwrap()
        .materialize()
        .unwrap();
    let elapsed = start.elapsed();

    assert!(result.nrows() > 0);
    assert_eq!(result.ncols(), 2);

    eprintln!(
        "[bench_filter_select_chain_100k] total: {:?}, rows: {}",
        elapsed, result.nrows()
    );
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

// ═══════════════════════════════════════════════════════════════════════════
// Group-by + Summarise benchmarks
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_group_summarise_100k_3_groups() {
    let df = make_large_df(100_000);
    let view = df.tidy();

    let start = Instant::now();
    let grouped = view.group_by(&["group"]).unwrap();
    let elapsed_group = start.elapsed();

    let start2 = Instant::now();
    let result = grouped.summarise(&[
        ("total", TidyAgg::Sum("value".into())),
        ("avg", TidyAgg::Mean("value".into())),
        ("n", TidyAgg::Count),
    ]).unwrap();
    let elapsed_agg = start2.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 3); // 3 groups

    eprintln!(
        "[bench_group_summarise_100k_3g] group: {:?}, summarise: {:?}",
        elapsed_group, elapsed_agg
    );
    assert!(elapsed_group.as_millis() + elapsed_agg.as_millis() < 5000);
}

#[test]
fn bench_group_summarise_100k_1000_groups() {
    let df = make_many_groups_df(100_000, 1000);
    let view = df.tidy();

    let start = Instant::now();
    let grouped = view.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("total", TidyAgg::Sum("value".into())),
        ("avg", TidyAgg::Mean("value".into())),
        ("cnt", TidyAgg::Count),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 1000);

    eprintln!(
        "[bench_group_summarise_100k_1000g] total: {:?}",
        elapsed
    );
    // Allow generous time in debug mode (release is ~915ms)
    assert!(elapsed.as_millis() < 120_000);
}

#[test]
fn bench_group_summarise_1m_100_groups() {
    let df = make_many_groups_df(1_000_000, 100);
    let view = df.tidy();

    let start = Instant::now();
    let grouped = view.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("total", TidyAgg::Sum("value".into())),
        ("avg", TidyAgg::Mean("value".into())),
        ("cnt", TidyAgg::Count),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 100);

    eprintln!(
        "[bench_group_summarise_1m_100g] total: {:?}",
        elapsed
    );
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

// ═══════════════════════════════════════════════════════════════════════════
// New TidyAgg benchmarks (Median, Sd, Var)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_group_advanced_aggs_100k() {
    let df = make_many_groups_df(100_000, 50);
    let view = df.tidy();

    let start = Instant::now();
    let grouped = view.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("med", TidyAgg::Median("value".into())),
        ("sd_val", TidyAgg::Sd("value".into())),
        ("var_val", TidyAgg::Var("value".into())),
        ("nd", TidyAgg::NDistinct("value".into())),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 50);

    eprintln!(
        "[bench_group_advanced_aggs_100k_50g] total: {:?}",
        elapsed
    );
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

// ═══════════════════════════════════════════════════════════════════════════
// Arrange (sort) benchmarks
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_arrange_100k_rows() {
    let df = make_large_df(100_000);
    let view = df.tidy();

    let start = Instant::now();
    let sorted = view.arrange(&[ArrangeKey::desc("value")]).unwrap();
    let mat = sorted.materialize().unwrap();
    let elapsed = start.elapsed();

    assert_eq!(mat.nrows(), 100_000);
    // Verify descending order
    if let Column::Float(vals) = mat.get_column("value").unwrap() {
        for i in 1..vals.len() {
            assert!(vals[i - 1] >= vals[i], "not sorted descending at index {}", i);
        }
    }

    eprintln!("[bench_arrange_100k] total: {:?}", elapsed);
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

// ═══════════════════════════════════════════════════════════════════════════
// DExpr window function benchmarks
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_cumsum_mutate_100k() {
    let df = make_large_df(100_000);
    let view = df.tidy();

    let start = Instant::now();
    let result = view.mutate(&[
        ("running_sum", DExpr::CumSum(Box::new(DExpr::Col("value".into())))),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 100_000);
    assert!(df_out.get_column("running_sum").is_some());

    eprintln!("[bench_cumsum_mutate_100k] total: {:?}", elapsed);
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

// ═══════════════════════════════════════════════════════════════════════════
// Dictionary encoding benchmark
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_dict_encoding_100k() {
    use cjc_data::dict_encoding::DictEncoding;

    let mut strings = Vec::with_capacity(100_000);
    let categories = ["cat_a", "cat_b", "cat_c", "cat_d", "cat_e",
                      "cat_f", "cat_g", "cat_h", "cat_i", "cat_j"];
    for i in 0..100_000 {
        strings.push(categories[i % 10].to_string());
    }

    let start = Instant::now();
    let encoded = DictEncoding::encode(&strings);
    let elapsed_encode = start.elapsed();

    assert_eq!(encoded.cardinality(), 10);
    assert_eq!(encoded.codes().len(), 100_000);

    let start2 = Instant::now();
    let decoded = encoded.decode();
    let elapsed_decode = start2.elapsed();

    assert_eq!(decoded, strings);

    eprintln!(
        "[bench_dict_encoding_100k] encode: {:?}, decode: {:?}, cardinality: {}",
        elapsed_encode, elapsed_decode, encoded.cardinality()
    );
    assert!(elapsed_encode.as_millis() + elapsed_decode.as_millis() < 5000);
}

// ═══════════════════════════════════════════════════════════════════════════
// Agg kernel benchmarks
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_agg_kernels_100k() {
    use cjc_data::agg_kernels;

    let data: Vec<f64> = (0..100_000).map(|i| i as f64 * 1.1).collect();
    // 100 segments of 1000 elements each
    let segments: Vec<(usize, usize)> = (0..100).map(|i| (i * 1000, (i + 1) * 1000)).collect();

    let start = Instant::now();
    let sums = agg_kernels::agg_sum_f64(&data, &segments);
    let means = agg_kernels::agg_mean_f64(&data, &segments);
    let vars = agg_kernels::agg_var_f64(&data, &segments);
    let medians = agg_kernels::agg_median_f64(&data, &segments);
    let elapsed = start.elapsed();

    assert_eq!(sums.len(), 100);
    assert_eq!(means.len(), 100);
    assert_eq!(vars.len(), 100);
    assert_eq!(medians.len(), 100);

    // Verify first segment sum: sum(0..1000) * 1.1
    let expected_sum: f64 = (0..1000).map(|i| i as f64 * 1.1).sum();
    assert!((sums[0] - expected_sum).abs() < 1.0); // within rounding

    eprintln!(
        "[bench_agg_kernels_100k] 100 segments x 1000 elements, 4 aggs: {:?}",
        elapsed
    );
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

// ═══════════════════════════════════════════════════════════════════════════
// Determinism verification on large data
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn determinism_large_pipeline_3_runs() {
    // Run the same pipeline 3 times, verify bit-identical output
    let mut results = Vec::new();

    for _ in 0..3 {
        let df = make_many_groups_df(10_000, 50);
        let view = df.tidy();
        let grouped = view.group_by(&["group"]).unwrap();
        let result = grouped.summarise(&[
            ("total", TidyAgg::Sum("value".into())),
            ("avg", TidyAgg::Mean("value".into())),
            ("med", TidyAgg::Median("value".into())),
            ("sd_val", TidyAgg::Sd("value".into())),
        ]).unwrap();
        let df_out = result.borrow();

        // Collect all values as bits for exact comparison
        let mut bits = Vec::new();
        for (name, col) in &df_out.columns {
            bits.push(name.clone());
            match col {
                Column::Float(vals) => {
                    for v in vals {
                        bits.push(format!("{}", v.to_bits()));
                    }
                }
                Column::Int(vals) => {
                    for v in vals {
                        bits.push(format!("{}", v));
                    }
                }
                Column::Str(vals) => {
                    for v in vals {
                        bits.push(v.clone());
                    }
                }
                _ => {}
            }
        }
        results.push(bits);
    }

    assert_eq!(results[0], results[1], "Run 1 != Run 2");
    assert_eq!(results[1], results[2], "Run 2 != Run 3");
}

// ═══════════════════════════════════════════════════════════════════════════
// Full pipeline benchmark: filter → select → group → summarise
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_full_pipeline_100k() {
    let df = make_large_df(100_000);
    let view = df.tidy();

    let start = Instant::now();

    // filter → select → group → summarise
    let filtered = view.filter(&DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(DExpr::Col("value".into())),
        right: Box::new(DExpr::LitFloat(25_000.0)),
    }).unwrap();

    let selected = filtered.select(&["group", "value"]).unwrap();
    let materialized = selected.materialize().unwrap();
    let view2 = materialized.tidy();
    let grouped = view2.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("total", TidyAgg::Sum("value".into())),
        ("avg", TidyAgg::Mean("value".into())),
        ("cnt", TidyAgg::Count),
    ]).unwrap();

    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 3);

    eprintln!(
        "[bench_full_pipeline_100k] filter→select→group→summarise: {:?}",
        elapsed
    );
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

#[test]
fn bench_full_pipeline_1m() {
    let df = make_large_df(1_000_000);
    let view = df.tidy();

    let start = Instant::now();

    let filtered = view.filter(&DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(DExpr::Col("value".into())),
        right: Box::new(DExpr::LitFloat(500_000.0)),
    }).unwrap();

    let mat = filtered.materialize().unwrap();
    let view2 = mat.tidy();
    let grouped = view2.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("total", TidyAgg::Sum("value".into())),
        ("avg", TidyAgg::Mean("value".into())),
        ("cnt", TidyAgg::Count),
    ]).unwrap();

    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 3);

    eprintln!(
        "[bench_full_pipeline_1m] filter→group→summarise: {:?}",
        elapsed
    );
    assert!(elapsed.as_millis() < 120_000); // generous for debug mode
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZATION-SPECIFIC BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

// ── O1: Fast group_by with BTreeMap ─────────────────────────────────────

#[test]
fn bench_o1_group_by_5000_groups_500k() {
    let df = make_many_groups_df(500_000, 5000);
    let view = df.tidy();

    let start = Instant::now();
    let grouped = view.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("total", TidyAgg::Sum("value".into())),
        ("cnt", TidyAgg::Count),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 5000);

    eprintln!("[bench_o1_group_5000g_500k] total: {:?}", elapsed);
    assert!(elapsed.as_secs() < 120);
}

// ── O3: Columnar predicate evaluation ───────────────────────────────────

#[test]
fn bench_o3_columnar_filter_1m() {
    let df = make_large_df(1_000_000);
    let view = df.tidy();

    let start = Instant::now();
    let filtered = view.filter(&DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(DExpr::Col("value".into())),
        right: Box::new(DExpr::LitFloat(500_000.0)),
    }).unwrap();
    let mat = filtered.materialize().unwrap();
    let elapsed = start.elapsed();

    assert!(mat.nrows() > 0);
    assert!(mat.nrows() < 1_000_000);

    eprintln!(
        "[bench_o3_columnar_filter_1m] total: {:?}, rows: 1M -> {}",
        elapsed, mat.nrows()
    );
    assert!(elapsed.as_secs() < 120);
}

#[test]
fn bench_o3_compound_filter_and_500k() {
    let df = make_large_df(500_000);
    let view = df.tidy();

    let pred = DExpr::BinOp {
        op: cjc_data::DBinOp::And,
        left: Box::new(DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(DExpr::Col("value".into())),
            right: Box::new(DExpr::LitFloat(100_000.0)),
        }),
        right: Box::new(DExpr::BinOp {
            op: cjc_data::DBinOp::Lt,
            left: Box::new(DExpr::Col("id".into())),
            right: Box::new(DExpr::LitInt(400_000)),
        }),
    };

    let start = Instant::now();
    let filtered = view.filter(&pred).unwrap();
    let mat = filtered.materialize().unwrap();
    let elapsed = start.elapsed();

    assert!(mat.nrows() > 0);
    eprintln!(
        "[bench_o3_compound_and_500k] total: {:?}, rows: 500K -> {}",
        elapsed, mat.nrows()
    );
    assert!(elapsed.as_secs() < 120);
}

// ── O5+O9: Fast segment aggregation + arena ─────────────────────────────

#[test]
fn bench_o5_fast_agg_median_sd_500k() {
    let df = make_many_groups_df(500_000, 100);
    let view = df.tidy();

    let start = Instant::now();
    let grouped = view.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("med", TidyAgg::Median("value".into())),
        ("sd_val", TidyAgg::Sd("value".into())),
        ("var_val", TidyAgg::Var("value".into())),
        ("iqr_val", TidyAgg::Iqr("value".into())),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 100);

    eprintln!("[bench_o5_fast_agg_median_sd_500k] total: {:?}", elapsed);
    assert!(elapsed.as_secs() < 120);
}

// ── O6: Join key caching ────────────────────────────────────────────────

#[test]
fn bench_o6_join_100k_x_1k() {
    let mut lk = Vec::with_capacity(100_000);
    let mut lv = Vec::with_capacity(100_000);
    for i in 0..100_000usize {
        lk.push(format!("k{}", i % 1000));
        lv.push(i as f64);
    }
    let left = DataFrame::from_columns(vec![
        ("key".into(), Column::Str(lk)),
        ("lval".into(), Column::Float(lv)),
    ]).unwrap();

    let mut rk = Vec::with_capacity(1000);
    let mut rv = Vec::with_capacity(1000);
    for i in 0..1000usize {
        rk.push(format!("k{}", i));
        rv.push(i as f64 * 100.0);
    }
    let right = DataFrame::from_columns(vec![
        ("key".into(), Column::Str(rk)),
        ("rval".into(), Column::Float(rv)),
    ]).unwrap();

    let left_view = left.tidy();
    let right_view = right.tidy();

    let start = Instant::now();
    let joined = left_view.inner_join(&right_view, &[("key", "key")]).unwrap();
    let mat_frame = joined.borrow();
    let elapsed = start.elapsed();

    assert!(mat_frame.nrows() > 0);
    eprintln!(
        "[bench_o6_join_100k_x_1k] total: {:?}, result rows: {}",
        elapsed, mat_frame.nrows()
    );
    assert!(elapsed.as_secs() < 120);
}

// ── O7: Vectorized DExpr ────────────────────────────────────────────────

#[test]
fn bench_o7_vectorized_mutate_500k() {
    let df = make_large_df(500_000);
    let view = df.tidy();

    let expr = DExpr::BinOp {
        op: cjc_data::DBinOp::Add,
        left: Box::new(DExpr::BinOp {
            op: cjc_data::DBinOp::Mul,
            left: Box::new(DExpr::Col("value".into())),
            right: Box::new(DExpr::LitFloat(2.0)),
        }),
        right: Box::new(DExpr::LitFloat(1.0)),
    };

    let start = Instant::now();
    let result = view.mutate(&[("doubled", expr)]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 500_000);
    assert!(df_out.get_column("doubled").is_some());

    eprintln!("[bench_o7_vectorized_mutate_500k] total: {:?}", elapsed);
    assert!(elapsed.as_secs() < 120);
}

// ── O8: BTreeSet distinct ───────────────────────────────────────────────

#[test]
fn bench_o8_distinct_500k_1000_unique() {
    let df = make_many_groups_df(500_000, 1000);
    let view = df.tidy();

    let start = Instant::now();
    let result = view.distinct(&["group"]).unwrap();
    let mat = result.materialize().unwrap();
    let elapsed = start.elapsed();

    assert_eq!(mat.nrows(), 1000);
    eprintln!("[bench_o8_distinct_500k_1000u] total: {:?}", elapsed);
    assert!(elapsed.as_secs() < 120);
}

// ── SQL-1+2: Lazy plan with optimizer ───────────────────────────────────

#[test]
fn bench_lazy_pipeline_100k() {
    let df = make_large_df(100_000);

    let start = Instant::now();
    let result = LazyView::from_df(df)
        .filter(DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(DExpr::Col("value".into())),
            right: Box::new(DExpr::LitFloat(25_000.0)),
        })
        .select(vec!["group".into(), "value".into()])
        .collect()
        .unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert!(df_out.nrows() > 0);
    assert_eq!(df_out.ncols(), 2);

    eprintln!(
        "[bench_lazy_pipeline_100k] filter->select: {:?}, rows: {}",
        elapsed, df_out.nrows()
    );
    assert!(elapsed.as_secs() < 120);
}

#[test]
fn bench_lazy_vs_eager_1m() {
    let df1 = make_large_df(1_000_000);
    let df2 = make_large_df(1_000_000);

    // Eager path
    let start_eager = Instant::now();
    let view = df1.tidy();
    let filtered = view.filter(&DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(DExpr::Col("value".into())),
        right: Box::new(DExpr::LitFloat(500_000.0)),
    }).unwrap();
    let selected = filtered.select(&["group", "value"]).unwrap();
    let _eager_result = selected.materialize().unwrap();
    let elapsed_eager = start_eager.elapsed();

    // Lazy path
    let start_lazy = Instant::now();
    let lazy_result = LazyView::from_df(df2)
        .filter(DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(DExpr::Col("value".into())),
            right: Box::new(DExpr::LitFloat(500_000.0)),
        })
        .select(vec!["group".into(), "value".into()])
        .collect()
        .unwrap();
    let elapsed_lazy = start_lazy.elapsed();

    let lazy_df = lazy_result.borrow();
    assert!(lazy_df.nrows() > 0);

    eprintln!(
        "[bench_lazy_vs_eager_1m] eager: {:?}, lazy: {:?}, ratio: {:.2}x",
        elapsed_eager, elapsed_lazy,
        elapsed_eager.as_micros() as f64 / elapsed_lazy.as_micros().max(1) as f64
    );
}

// ── SQL-3: Batch execution ──────────────────────────────────────────────

#[test]
fn bench_batch_vs_standard_500k() {
    let df1 = make_large_df(500_000);
    let df2 = make_large_df(500_000);

    let start_std = Instant::now();
    let std_result = LazyView::from_df(df1)
        .filter(DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(DExpr::Col("value".into())),
            right: Box::new(DExpr::LitFloat(250_000.0)),
        })
        .select(vec!["group".into(), "value".into()])
        .collect()
        .unwrap();
    let elapsed_std = start_std.elapsed();

    let start_batch = Instant::now();
    let batch_result = LazyView::from_df(df2)
        .filter(DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(DExpr::Col("value".into())),
            right: Box::new(DExpr::LitFloat(250_000.0)),
        })
        .select(vec!["group".into(), "value".into()])
        .collect_batched()
        .unwrap();
    let elapsed_batch = start_batch.elapsed();

    let std_df = std_result.borrow();
    let batch_df = batch_result.borrow();
    assert_eq!(std_df.nrows(), batch_df.nrows());

    eprintln!(
        "[bench_batch_vs_standard_500k] standard: {:?}, batched: {:?}, ratio: {:.2}x",
        elapsed_std, elapsed_batch,
        elapsed_std.as_micros() as f64 / elapsed_batch.as_micros().max(1) as f64
    );
}

// ── SQL-4+5: Zone maps + sorted column detection ────────────────────────

#[test]
fn bench_zone_maps_1m() {
    use cjc_data::column_meta::DataFrameStats;

    let df = make_large_df(1_000_000);

    let start = Instant::now();
    let stats = DataFrameStats::compute(&df);
    let elapsed_compute = start.elapsed();

    let val_stats = stats.get("value").unwrap();
    assert!(!val_stats.can_skip_gt(0.0));
    assert!(val_stats.can_skip_gt(2_000_000.0));

    let id_stats = stats.get("id").unwrap();
    assert!(id_stats.sorted_asc, "id column should be detected as sorted ascending");

    eprintln!(
        "[bench_zone_maps_1m] compute stats for 1M rows: {:?}",
        elapsed_compute
    );
    assert!(elapsed_compute.as_secs() < 120);
}

// ── SQL-6: Rolling window aggregation ───────────────────────────────────

#[test]
fn bench_rolling_sum_500k() {
    let df = make_large_df(500_000);
    let view = df.tidy();

    let start = Instant::now();
    let result = view.mutate(&[
        ("roll_sum", DExpr::RollingSum("value".into(), 100)),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 500_000);
    assert!(df_out.get_column("roll_sum").is_some());

    eprintln!("[bench_rolling_sum_500k_w100] total: {:?}", elapsed);
    assert!(elapsed.as_secs() < 120);
}

#[test]
fn bench_rolling_min_max_500k() {
    let df = make_large_df(500_000);
    let view = df.tidy();

    let start = Instant::now();
    let result = view.mutate(&[
        ("roll_min", DExpr::RollingMin("value".into(), 50)),
        ("roll_max", DExpr::RollingMax("value".into(), 50)),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 500_000);

    eprintln!("[bench_rolling_min_max_500k_w50] total: {:?}", elapsed);
    assert!(elapsed.as_secs() < 120);
}

// ═══════════════════════════════════════════════════════════════════════════
// EXTREME SCALE BENCHMARKS (5M rows)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_extreme_filter_5m() {
    let df = make_large_df(5_000_000);
    let view = df.tidy();

    let start = Instant::now();
    let filtered = view.filter(&DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(DExpr::Col("value".into())),
        right: Box::new(DExpr::LitFloat(2_750_000.0)),
    }).unwrap();
    let mat = filtered.materialize().unwrap();
    let elapsed = start.elapsed();

    assert!(mat.nrows() > 0);
    eprintln!(
        "[bench_extreme_filter_5m] total: {:?}, rows: 5M -> {}",
        elapsed, mat.nrows()
    );
    assert!(elapsed.as_secs() < 300);
}

#[test]
fn bench_extreme_group_5m_1000g() {
    let df = make_many_groups_df(5_000_000, 1000);
    let view = df.tidy();

    let start = Instant::now();
    let grouped = view.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("total", TidyAgg::Sum("value".into())),
        ("avg", TidyAgg::Mean("value".into())),
        ("cnt", TidyAgg::Count),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 1000);

    eprintln!("[bench_extreme_group_5m_1000g] total: {:?}", elapsed);
    assert!(elapsed.as_secs() < 300);
}

#[test]
fn bench_extreme_full_pipeline_5m() {
    let df = make_large_df(5_000_000);
    let view = df.tidy();

    let start = Instant::now();
    let filtered = view.filter(&DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(DExpr::Col("value".into())),
        right: Box::new(DExpr::LitFloat(1_000_000.0)),
    }).unwrap();
    let mat = filtered.materialize().unwrap();
    let view2 = mat.tidy();
    let grouped = view2.group_by(&["group"]).unwrap();
    let result = grouped.summarise(&[
        ("total", TidyAgg::Sum("value".into())),
        ("avg", TidyAgg::Mean("value".into())),
        ("cnt", TidyAgg::Count),
    ]).unwrap();
    let elapsed = start.elapsed();

    let df_out = result.borrow();
    assert_eq!(df_out.nrows(), 3);

    eprintln!(
        "[bench_extreme_pipeline_5m] filter->group->summarise: {:?}",
        elapsed
    );
    assert!(elapsed.as_secs() < 300);
}
