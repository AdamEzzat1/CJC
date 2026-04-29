// Adaptive TidyView Engine v2 — density crossover benchmark
//
// Sweeps result density from 0% → 100% in ~5% steps and reports:
//   - selection mode chosen
//   - filter wall time
//   - downstream iter_indices() wall time
//
// Run with:
//   cargo test --test test_phase10_tidy adaptive_selection_bench --release -- --ignored --nocapture
//
// This is a *design-validation* benchmark, not a regression gate. It's not
// hooked into CI. If a future change makes one of the modes regress, run
// this manually and update the published crossover figures in the docs.

use cjc_data::{Column, DBinOp, DExpr, DataFrame};
use std::time::Instant;

fn pred_lt(col: &str, v: i64) -> DExpr {
    DExpr::BinOp {
        op: DBinOp::Lt,
        left: Box::new(DExpr::Col(col.into())),
        right: Box::new(DExpr::LitInt(v)),
    }
}

fn make_df(n: usize) -> DataFrame {
    let xs: Vec<i64> = (0..n as i64).collect();
    DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap()
}

#[test]
#[ignore = "design-validation benchmark; run manually with --ignored --nocapture"]
fn bench_density_crossover() {
    const N: usize = 1_000_000;
    let df = make_df(N);

    println!();
    println!("Adaptive TidyView Engine v2 — density crossover bench");
    println!("nrows = {N}");
    println!(
        "{:>12} {:>10} {:>18} {:>14} {:>14}",
        "threshold", "hits", "mode", "filter_us", "iter_us"
    );
    println!("{:->76}", "");

    // Absolute thresholds first (sparse regime: 1, 10, 100, 1000 hits — exercises
    // the SelectionVector arm which percentages can't reach when N=1M and the
    // sparse cutoff is N/1024 = 976), then percentage sweep for mid + dense.
    let mut thresholds: Vec<i64> = vec![0, 1, 10, 100, 1000];
    for pct in [1u64, 5, 10, 20, 30, 40, 50, 70, 90, 99, 100] {
        thresholds.push((N as u64 * pct / 100) as i64);
    }

    for threshold in thresholds {
        let predicate = pred_lt("x", threshold);

        let t0 = Instant::now();
        let v = df.clone().tidy().filter(&predicate).unwrap();
        let filter_us = t0.elapsed().as_micros();

        let t1 = Instant::now();
        let collected: Vec<usize> = v.selection().iter_indices().collect();
        let iter_us = t1.elapsed().as_micros();

        let mode = v.explain_selection_mode();
        println!(
            "{:>12} {:>10} {:>18} {:>14} {:>14}",
            threshold,
            collected.len(),
            mode,
            filter_us,
            iter_us
        );
    }
    println!();
}

/// v2.2 chain bench — measures the wall-clock effect of the sparse-aware
/// predicate bytecode path on chained `.filter().filter()` workloads.
///
/// The first filter narrows to ~0.01% of rows (well below the 25% sparse
/// threshold). The second filter is observed twice: once with `count`
/// already in the sparse band (current v2.2 behavior — random-access
/// gather over `iter_indices()`), and once would-have-been with the
/// dense path (we report the projected dense cost from the column scan
/// of the standalone-filter bench above for reference, but this is
/// labeled clearly).
///
/// Run with:
///   cargo test --test test_phase10_tidy bench_sparse_chain --release -- --ignored --nocapture
#[test]
#[ignore = "design-validation benchmark; run manually with --ignored --nocapture"]
fn bench_sparse_chain() {
    const N: usize = 1_000_000;
    let df = make_df(N);

    println!();
    println!("Adaptive TidyView Engine v2.2 — sparse-chain bench");
    println!("nrows = {N}");
    println!(
        "{:>12} {:>10} {:>14} {:>14} {:>18}",
        "p1_thresh", "p1_hits", "step1_us", "step2_us", "step2_mode"
    );
    println!("{:->76}", "");

    // p1 narrows to {1, 10, 100, 1000, 10_000, 100_000} hits — first three
    // land deep in sparse-path territory; 10k and 100k cross the threshold;
    // 100k is exactly at 10% (sparse path still picks it because count*4 < N).
    let p1_thresholds: [i64; 6] = [1, 10, 100, 1000, 10_000, 100_000];

    for p1_t in p1_thresholds {
        let p1 = pred_lt("x", p1_t);
        // p2 keeps the lower half of whatever p1 produced.
        let p2 = pred_lt("x", p1_t / 2);

        let t0 = Instant::now();
        let v1 = df.clone().tidy().filter(&p1).unwrap();
        let step1_us = t0.elapsed().as_micros();

        let p1_hits = v1.nrows();

        let t1 = Instant::now();
        let v2 = v1.filter(&p2).unwrap();
        let step2_us = t1.elapsed().as_micros();

        let mode = v2.explain_selection_mode();
        println!(
            "{:>12} {:>10} {:>14} {:>14} {:>18}",
            p1_t, p1_hits, step1_us, step2_us, mode
        );
    }
    println!();
}

/// Smoke version of the bench — runs in CI to confirm the bench scaffold
/// itself does not regress (e.g. all four modes are reachable).
#[test]
fn bench_density_crossover_smoke() {
    const N: usize = 100_000;
    let df = make_df(N);

    // Absolute thresholds chosen to land each adaptive arm:
    //   0      -> Empty
    //   50     -> SelectionVector (sparse, < N/1024 = 97)
    //   5_000  -> Hybrid          (mid-band, 5% of N, large frame)
    //   50_000 -> VerbatimMask    (dense, 50% > 30% threshold)
    //   N      -> All             (every row passes)
    let mut modes_seen = std::collections::BTreeSet::new();
    for threshold in [0i64, 50, 5_000, 50_000, N as i64] {
        let v = df.clone().tidy().filter(&pred_lt("x", threshold)).unwrap();
        modes_seen.insert(v.explain_selection_mode().to_string());
    }
    assert!(modes_seen.contains("Empty"), "Empty mode unreachable");
    assert!(modes_seen.contains("All"), "All mode unreachable");
    assert!(
        modes_seen.contains("SelectionVector"),
        "SelectionVector mode unreachable"
    );
    assert!(
        modes_seen.contains("VerbatimMask"),
        "VerbatimMask mode unreachable"
    );
    assert!(
        modes_seen.contains("Hybrid"),
        "Hybrid mode unreachable in v2.1 — mid-band classifier may not be active"
    );
}
