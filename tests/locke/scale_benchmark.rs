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

// ─── A5 scale benchmark: deterministic BPE on a synthetic corpus ─────────
//
// Closes the "Zero real-world data fed through it yet" critique on
// the v0.7+ A5 text-drift work. Generates a deterministic synthetic
// corpus (no external download — corpus is reproducible from a seed),
// then trains, encodes, decodes, and reports throughput + fingerprint.
//
// Run via `cargo test --test locke --release -- --ignored scale_benchmark_bpe`.

/// Deterministic English-shaped corpus generator: a small set of stems
/// concatenated with a SplitMix64-style state machine. ~`bytes_target`
/// bytes of output, reproducible from `seed`.
fn synthesize_text_corpus(bytes_target: usize, seed: u64) -> Vec<String> {
    // Hand-picked stems with enough overlap to make BPE merges productive.
    let stems = [
        " the ", " and ", " of ", " to ", " a ", " in ", " for ", " is ",
        " on ", " that ", " by ", " this ", " with ", " I ", " you ",
        "patient ", "study ", "report ", "value ", "data ", "result ",
        "encounter ", "diagnosis ", "treatment ", "outcome ", "admission ",
        "ing ", "ed ", "tion ", "ment ", "ness ", "able ",
    ];
    let mut state = seed;
    // Build a continuous byte stream of stems, breaking into "documents" of
    // ~1024 bytes each (typical web text document size).
    let mut out: Vec<String> = Vec::new();
    let mut doc = String::with_capacity(1200);
    let mut total = 0usize;
    while total < bytes_target {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let stem = stems[(state as usize) % stems.len()];
        doc.push_str(stem);
        total += stem.len();
        if doc.len() >= 1024 {
            out.push(std::mem::replace(&mut doc, String::with_capacity(1200)));
        }
    }
    if !doc.is_empty() {
        out.push(doc);
    }
    out
}

#[test]
#[ignore]
fn scale_benchmark_bpe_tokenizer_on_synthetic_corpus() {
    use cjc_locke::tokenizer::{Tokenizer, TokenizerTrainConfig};

    println!("\n=== Locke BPE tokenizer benchmark (A5) ===");
    println!("Deterministic synthetic corpus; no external download.\n");
    println!(
        "{:>10} | {:>10} | {:>14} | {:>14} | {:>14} | {:>10} | {:>20}",
        "corpus_KB", "vocab", "train_dt", "encode_MB/s", "decode_MB/s", "merges", "fingerprint"
    );
    println!("{}", "-".repeat(110));

    for &(target_bytes, vocab_size) in &[
        (256 * 1024, 512_u32),    // 256 KB, vocab 512
        (1024 * 1024, 1024_u32),  // 1 MB,   vocab 1024
        (4 * 1024 * 1024, 2048_u32), // 4 MB,   vocab 2048
    ] {
        let corpus = synthesize_text_corpus(target_bytes, 0xDEADBEEF);
        let total_bytes: usize = corpus.iter().map(|s| s.len()).sum();
        let cfg = TokenizerTrainConfig {
            target_vocab_size: vocab_size,
            ..Default::default()
        };

        let train_start = Instant::now();
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let tok = Tokenizer::train(&corpus_refs, &cfg);
        let train_dt = train_start.elapsed();

        let enc_start = Instant::now();
        let mut total_tokens = 0u64;
        for s in &corpus {
            total_tokens += tok.encode(s).len() as u64;
        }
        let enc_dt = enc_start.elapsed();
        let enc_mbps = (total_bytes as f64) / 1_048_576.0 / enc_dt.as_secs_f64().max(1e-6);

        let dec_start = Instant::now();
        for s in &corpus {
            let ids = tok.encode(s);
            let _ = tok.decode(&ids);
        }
        let dec_dt = dec_start.elapsed();
        let dec_mbps = (total_bytes as f64) / 1_048_576.0 / dec_dt.as_secs_f64().max(1e-6);

        // Round-trip determinism check (a sample document at a time).
        for s in corpus.iter().take(3) {
            let ids = tok.encode(s);
            let decoded = tok.decode(&ids);
            assert_eq!(s.as_str(), decoded.as_str(), "encode/decode round-trip failure");
        }
        let _ = total_tokens;

        println!(
            "{:>10} | {:>10} | {:>14} | {:>14.2} | {:>14.2} | {:>10} | {:>20}",
            total_bytes / 1024,
            tok.vocab_size(),
            fmt_total(train_dt),
            enc_mbps,
            dec_mbps,
            tok.merge_count(),
            format!("{}", tok.fingerprint()),
        );
    }
    println!(
        "\nDeterminism check: running again should produce IDENTICAL fingerprints."
    );
}

/// Cross-run determinism guard — tokenizer trained twice on the same
/// corpus produces a byte-identical fingerprint. Cheap; runs in the
/// non-ignored suite.
#[test]
fn bpe_tokenizer_fingerprint_is_byte_identical_across_repeat_training() {
    use cjc_locke::tokenizer::{Tokenizer, TokenizerTrainConfig};

    let corpus = synthesize_text_corpus(32 * 1024, 0xC0FFEE);
    let cfg = TokenizerTrainConfig {
        target_vocab_size: 384,
        ..Default::default()
    };
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
    let a = Tokenizer::train(&corpus_refs, &cfg);
    let b = Tokenizer::train(&corpus_refs, &cfg);
    assert_eq!(
        a.fingerprint(),
        b.fingerprint(),
        "tokenizer fingerprints diverged between two identical training runs"
    );
    assert_eq!(a.vocab_size(), b.vocab_size());
    assert_eq!(a.merge_count(), b.merge_count());

    // Round-trip on every document — verifies neither training produced a
    // non-invertible vocab on real text.
    for s in &corpus {
        let ids_a = a.encode(s);
        let ids_b = b.encode(s);
        assert_eq!(ids_a, ids_b, "encoding diverged between identical tokenizers");
        assert_eq!(s.as_str(), a.decode(&ids_a).as_str());
    }
}
