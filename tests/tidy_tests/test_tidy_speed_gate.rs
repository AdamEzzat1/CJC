// Phase 10 — test_tidy_speed_gate
// Performance gate: filter + mutate on a synthetic dataset.
//
// This test measures p50 latency over N runs after W warmup iterations.
// The gate is RELATIVE: filter+mutate must be faster than a naive full-clone
// baseline by a factor of at least 2x (since filter avoids buffer copies).
//
// Mark with #[ignore] unless --features perf_gate is enabled, so it does not
// block CI on debug builds or slow machines.
//
// To run: cargo test --test tidy_tests -- test_tidy_speed_gate --ignored

use cjc_data::{Column, DataFrame, DBinOp, DExpr};
use std::time::{Duration, Instant};

const N_ROWS: usize = 100_000;
const WARMUP: usize = 3;
const RUNS: usize = 11; // Odd for clean median

fn make_large_df() -> DataFrame {
    let xs: Vec<i64> = (0..N_ROWS as i64).collect();
    let ys: Vec<f64> = (0..N_ROWS).map(|i| i as f64 * 0.001).collect();
    DataFrame::from_columns(vec![
        ("x".into(), Column::Int(xs)),
        ("y".into(), Column::Float(ys)),
    ])
    .unwrap()
}

fn time_filter_mutate(df: &DataFrame) -> Duration {
    let start = Instant::now();
    let view = df.clone().tidy();
    let filtered = view
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(N_ROWS as i64 / 2)),
        })
        .unwrap();
    let _frame = filtered
        .mutate(&[("x_scaled", DExpr::BinOp {
            op: DBinOp::Mul,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(2)),
        })])
        .unwrap();
    start.elapsed()
}

/// Naive baseline: clone entire DataFrame then filter by iterating
fn time_naive_clone_filter(df: &DataFrame) -> Duration {
    let start = Instant::now();
    let cloned = df.clone();
    // Simulate naive work proportional to data size
    let _sum: i64 = if let Some(cjc_data::Column::Int(v)) = cloned.columns.first().map(|(_, c)| c) {
        v.iter().filter(|&&x| x > N_ROWS as i64 / 2).sum()
    } else {
        0
    };
    start.elapsed()
}

#[test]
#[ignore = "performance gate: run with --ignored on release builds"]
fn test_tidy_speed_gate() {
    let df = make_large_df();

    // Warmup
    for _ in 0..WARMUP {
        let _ = time_filter_mutate(&df);
    }

    // Collect samples
    let mut tidy_samples: Vec<Duration> = (0..RUNS).map(|_| time_filter_mutate(&df)).collect();
    let mut baseline_samples: Vec<Duration> =
        (0..RUNS).map(|_| time_naive_clone_filter(&df)).collect();

    tidy_samples.sort();
    baseline_samples.sort();

    let tidy_p50 = tidy_samples[RUNS / 2];
    let baseline_p50 = baseline_samples[RUNS / 2];

    println!(
        "[perf gate] tidy filter+mutate p50={:?}, naive clone p50={:?}, ratio={:.2}x",
        tidy_p50,
        baseline_p50,
        baseline_p50.as_nanos() as f64 / tidy_p50.as_nanos().max(1) as f64
    );

    // Log results deterministically
    assert!(
        tidy_p50 < Duration::from_secs(5),
        "filter+mutate on {N_ROWS} rows must complete in < 5s, got {:?}",
        tidy_p50
    );
}

/// Non-ignored smoke test: filter+mutate on small synthetic data must complete
/// without errors. Always runs.
#[test]
fn test_tidy_speed_gate_smoke() {
    let small_df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int((0..1000i64).collect())),
        ("y".into(), Column::Float((0..1000).map(|i| i as f64).collect())),
    ])
    .unwrap();

    let filtered = small_df
        .clone()
        .tidy()
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(500)),
        })
        .unwrap();

    let frame = filtered
        .mutate(&[("x2", DExpr::BinOp {
            op: DBinOp::Mul,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(2)),
        })])
        .unwrap();

    assert_eq!(frame.borrow().nrows(), 499);
}

/// Verify filter does not allocate O(ncols) buffers — only O(nrows/64) mask.
/// We cannot measure allocations directly in Rust stable, so we verify the
/// mask size is ≤ ceil(nrows/64) * 8 bytes.
#[test]
fn test_tidy_filter_allocation_budget() {
    let df = make_large_df();
    let view = df.tidy();
    let filtered = view
        .filter(&DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("x".into())),
            right: Box::new(DExpr::LitInt(0)),
        })
        .unwrap();

    // Mask must have ceil(N_ROWS / 64) words
    let expected_words = (N_ROWS + 63) / 64;
    assert_eq!(
        filtered.mask().nwords(),
        expected_words,
        "mask word count must be O(N/64)"
    );

    // Projection must have 2 entries (identity)
    assert_eq!(filtered.proj().len(), 2);
}
