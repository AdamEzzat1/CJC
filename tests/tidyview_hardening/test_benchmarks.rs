//! TidyView performance benchmarks on large data.
//! Tests correctness AND measures wall-clock time for key operations.

use cjc_data::{Column, DataFrame, DExpr, TidyAgg, ArrangeKey};
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
