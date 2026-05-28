//! Scale benchmark for Locke. Run with:
//!
//! ```bash
//! cargo test --release --test locke scale_benchmark -- --ignored --nocapture
//! ```
//!
//! Reports wall-clock time and per-row throughput for:
//! - single-shot `validate(&df, &opts)`
//! - `compare(&train, &test)` drift
//! - `StreamingValidator` chunked ingest
//!
//! Synthetic data: 4 columns (Float, Int, Str, Bool) with seeded NaN,
//! duplicates, outliers, and sentinel values so every validator fires.

use cjc_data::{Column, DataFrame};
use cjc_locke::api::{validate, ValidateOptions};
use cjc_locke::drift::{compare, DriftConfig};
use cjc_locke::streaming::{StreamingConfig, StreamingValidator};
use cjc_repro::Rng;
use std::time::Instant;

fn synthesize(n: usize, seed: u64) -> DataFrame {
    let mut rng = Rng::seeded(seed);
    let mut float_col: Vec<f64> = Vec::with_capacity(n);
    let mut int_col: Vec<i64> = Vec::with_capacity(n);
    let mut str_col: Vec<String> = Vec::with_capacity(n);
    let mut bool_col: Vec<bool> = Vec::with_capacity(n);

    let categories = ["us", "uk", "de", "fr", "jp"];
    for i in 0..n {
        let r = rng.next_u64();
        // 2% NaN
        let f = if r % 50 == 0 {
            f64::NAN
        } else {
            (i as f64) * 0.5 + ((r % 1000) as f64) * 0.01
        };
        float_col.push(f);
        // 0.1% sentinels
        int_col.push(if r % 1000 == 0 { -9999 } else { (i as i64) % 100_000 });
        str_col.push(categories[(r % 5) as usize].into());
        bool_col.push((r % 2) == 0);
    }
    // Force ~0.5% duplicates by copying earlier rows.
    let n_dups = n / 200;
    for d in 0..n_dups.min(50) {
        if 2 * d < n {
            float_col[n - 1 - d] = float_col[d];
            int_col[n - 1 - d] = int_col[d];
            str_col[n - 1 - d] = str_col[d].clone();
            bool_col[n - 1 - d] = bool_col[d];
        }
    }

    DataFrame::from_columns(vec![
        ("amount".into(), Column::Float(float_col)),
        ("user_id".into(), Column::Int(int_col)),
        ("country".into(), Column::Str(str_col)),
        ("active".into(), Column::Bool(bool_col)),
    ])
    .unwrap()
}

fn fmt_per_row(elapsed_ns: u128, n: usize) -> String {
    let ns_per_row = elapsed_ns as f64 / n as f64;
    if ns_per_row < 1000.0 {
        format!("{:.1} ns/row", ns_per_row)
    } else if ns_per_row < 1_000_000.0 {
        format!("{:.2} us/row", ns_per_row / 1000.0)
    } else {
        format!("{:.2} ms/row", ns_per_row / 1_000_000.0)
    }
}

fn fmt_total(d: std::time::Duration) -> String {
    let ms = d.as_millis();
    if ms < 1000 {
        format!("{} ms", ms)
    } else {
        format!("{:.2} s", d.as_secs_f64())
    }
}

#[test]
#[ignore]
fn scale_benchmark_single_shot() {
    println!("\n=== Locke single-shot validate() benchmark ===");
    println!("Hardware: cargo bench-style, --release\n");
    println!("{:>12} | {:>12} | {:>14} | {:>4}", "rows", "wall_clock", "per_row", "findings");
    println!("{}", "-".repeat(62));

    for &n in &[10_000usize, 100_000, 1_000_000] {
        let df = synthesize(n, 0xCAFE);
        let opts = ValidateOptions {
            dataset_label: format!("bench-{}", n),
            ..Default::default()
        };
        let t = Instant::now();
        let r = validate(&df, &opts);
        let dt = t.elapsed();
        println!(
            "{:>12} | {:>12} | {:>14} | {:>4}",
            n,
            fmt_total(dt),
            fmt_per_row(dt.as_nanos(), n),
            r.findings.len()
        );
    }
}

#[test]
#[ignore]
fn scale_benchmark_drift_compare() {
    println!("\n=== Locke drift compare() benchmark ===\n");
    println!("{:>12} | {:>12} | {:>14} | {:>4}", "rows/side", "wall_clock", "per_row", "findings");
    println!("{}", "-".repeat(62));

    for &n in &[10_000usize, 100_000, 1_000_000] {
        let train = synthesize(n, 0x1111);
        let test = synthesize(n, 0x2222);
        let cfg = DriftConfig::default();
        let t = Instant::now();
        let r = compare(&train, &test, &cfg);
        let dt = t.elapsed();
        println!(
            "{:>12} | {:>12} | {:>14} | {:>4}",
            n,
            fmt_total(dt),
            fmt_per_row(dt.as_nanos(), 2 * n),
            r.findings.len()
        );
    }
}

#[test]
#[ignore]
fn scale_benchmark_streaming() {
    println!("\n=== Locke StreamingValidator benchmark ===\n");
    println!(
        "{:>12} | {:>8} | {:>12} | {:>14}",
        "total rows", "chunks", "wall_clock", "per_row"
    );
    println!("{}", "-".repeat(60));

    for &n in &[100_000usize, 1_000_000, 5_000_000] {
        let chunk_size = 10_000;
        let n_chunks = (n + chunk_size - 1) / chunk_size;
        let mut sv = StreamingValidator::new("bench", StreamingConfig::default());
        let t = Instant::now();
        for ci in 0..n_chunks {
            let lo = ci * chunk_size;
            let hi = (lo + chunk_size).min(n);
            let chunk = synthesize(hi - lo, 0x3333 + ci as u64);
            sv.ingest_chunk(&chunk).unwrap();
        }
        let dt = t.elapsed();
        let _summaries = sv.streaming_summaries();
        println!(
            "{:>12} | {:>8} | {:>12} | {:>14}",
            n,
            n_chunks,
            fmt_total(dt),
            fmt_per_row(dt.as_nanos(), n)
        );
    }
}

#[test]
#[ignore]
fn scale_benchmark_streaming_ks_d() {
    println!("\n=== Locke streaming_ks_d() vs single-shot KS benchmark ===\n");
    println!(
        "{:>12} | {:>20} | {:>20}",
        "rows", "streaming KS", "single-shot KS"
    );
    println!("{}", "-".repeat(58));

    for &n in &[10_000usize, 100_000, 1_000_000] {
        // Train: uniform [0, 1]
        let train_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        // Reference: same data, shifted by 0.5
        let mut reference: Vec<f64> =
            (0..n).map(|i| (i as f64) / (n as f64 - 1.0) + 0.5).collect();
        reference.sort_by(|a, b| a.total_cmp(b));

        // Streaming side
        let mut sv = StreamingValidator::new("ks", StreamingConfig::default());
        let chunk = DataFrame::from_columns(vec![("x".into(), Column::Float(train_vals.clone()))]).unwrap();
        sv.ingest_chunk(&chunk).unwrap();
        let t1 = Instant::now();
        let _d_stream = sv.streaming_ks_d("x", &reference).unwrap();
        let dt1 = t1.elapsed();

        // Single-shot side
        let t2 = Instant::now();
        let _d_single = cjc_locke::stats::ks_d_statistic(&train_vals, &reference).unwrap();
        let dt2 = t2.elapsed();

        println!(
            "{:>12} | {:>20} | {:>20}",
            n,
            fmt_total(dt1),
            fmt_total(dt2)
        );
    }
}
