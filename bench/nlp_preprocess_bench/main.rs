//! CJC NLP/ETL Preprocess Benchmark Suite (ROLE 7)
//! ================================================
//! Measures throughput and tail latency for the three reference kernels:
//!   A) Tokenizer — split_byte + trim_ascii at scale
//!   B) Vocab Counter — token counting + deterministic sort
//!   C) CSV Scanner — row/field splitting + validation
//!
//! Outputs JSONL to stdout for CI ingestion.
//! Prints a human-readable scorecard to stderr.

use cjc_ast::*;
use cjc_eval::*;
use cjc_runtime::{murmurhash3, value_hash, Value};
use std::time::Instant;

// -- AST Helpers (same as test harness) ------------------------------------

fn span() -> Span {
    Span::dummy()
}
fn ident(name: &str) -> Ident {
    Ident::dummy(name)
}
fn byte_string_expr(bytes: &[u8]) -> Expr {
    Expr {
        kind: ExprKind::ByteStringLit(bytes.to_vec()),
        span: span(),
    }
}
fn byte_char_expr(b: u8) -> Expr {
    Expr {
        kind: ExprKind::ByteCharLit(b),
        span: span(),
    }
}
fn call(callee: Expr, args: Vec<Expr>) -> Expr {
    let call_args: Vec<CallArg> = args
        .into_iter()
        .map(|value| CallArg {
            name: None,
            value,
            span: span(),
        })
        .collect();
    Expr {
        kind: ExprKind::Call {
            callee: Box::new(callee),
            args: call_args,
        },
        span: span(),
    }
}
fn field_expr(object: Expr, name: &str) -> Expr {
    Expr {
        kind: ExprKind::Field {
            object: Box::new(object),
            name: ident(name),
        },
        span: span(),
    }
}
fn method_call(object: Expr, method: &str, args: Vec<Expr>) -> Expr {
    call(field_expr(object, method), args)
}

fn eval_expr_val(expr: &Expr) -> Value {
    let mut interp = Interpreter::new(42);
    interp.eval_expr(expr).unwrap()
}

// -- Timing utilities -------------------------------------------------------

struct TimingSample {
    iterations: usize,
    total_ns: u128,
    per_iter_ns: Vec<u128>,
}

fn bench_iterations<F: FnMut()>(mut f: F, iters: usize) -> TimingSample {
    let mut per_iter_ns = Vec::with_capacity(iters);
    let total_start = Instant::now();
    for _ in 0..iters {
        let start = Instant::now();
        f();
        per_iter_ns.push(start.elapsed().as_nanos());
    }
    let total_ns = total_start.elapsed().as_nanos();
    TimingSample {
        iterations: iters,
        total_ns,
        per_iter_ns,
    }
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_jsonl(kernel: &str, input_size: usize, sample: &TimingSample) {
    let mut sorted = sample.per_iter_ns.clone();
    sorted.sort();
    let median = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);
    let p99 = percentile(&sorted, 99.0);
    let min = sorted.first().copied().unwrap_or(0);
    let max = sorted.last().copied().unwrap_or(0);
    let mean = sample.total_ns / sample.iterations as u128;

    // Throughput: bytes/sec based on input_size and mean time
    let throughput_mbs = if mean > 0 {
        (input_size as f64) / (mean as f64 / 1_000_000_000.0) / (1024.0 * 1024.0)
    } else {
        0.0
    };

    println!(
        r#"{{"kernel":"{}","input_bytes":{},"iterations":{},"mean_ns":{},"median_ns":{},"p95_ns":{},"p99_ns":{},"min_ns":{},"max_ns":{},"throughput_mbs":{:.2}}}"#,
        kernel,
        input_size,
        sample.iterations,
        mean,
        median,
        p95,
        p99,
        min,
        max,
        throughput_mbs
    );
}

fn print_scorecard_line(
    kernel: &str,
    input_size: usize,
    sample: &TimingSample,
) {
    let mut sorted = sample.per_iter_ns.clone();
    sorted.sort();
    let median = percentile(&sorted, 50.0);
    let p99 = percentile(&sorted, 99.0);
    let mean = sample.total_ns / sample.iterations as u128;
    let throughput_mbs = if mean > 0 {
        (input_size as f64) / (mean as f64 / 1_000_000_000.0) / (1024.0 * 1024.0)
    } else {
        0.0
    };

    eprintln!(
        "  {:20} {:>8} bytes  median: {:>8} ns  p99: {:>8} ns  {:.1} MB/s",
        kernel, input_size, median, p99, throughput_mbs
    );
}

// -- Data generators --------------------------------------------------------

fn generate_text(num_words: usize) -> Vec<u8> {
    // Deterministic pseudo-random word generation using murmurhash3
    let words: &[&[u8]] = &[
        b"the", b"quick", b"brown", b"fox", b"jumps", b"over", b"lazy", b"dog",
        b"cat", b"sat", b"on", b"mat", b"and", b"ran", b"fast", b"slow",
        b"big", b"small", b"red", b"blue", b"green", b"white", b"black", b"dark",
        b"light", b"warm", b"cold", b"hot", b"dry", b"wet", b"old", b"new",
    ];
    let mut buf = Vec::with_capacity(num_words * 6);
    for i in 0..num_words {
        if i > 0 {
            buf.push(b' ');
        }
        let h = murmurhash3(&(i as u64).to_le_bytes());
        let idx = (h as usize) % words.len();
        buf.extend_from_slice(words[idx]);
    }
    buf
}

fn generate_csv(num_rows: usize, num_cols: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(num_rows * num_cols * 8);
    // Header
    for c in 0..num_cols {
        if c > 0 {
            buf.push(b',');
        }
        buf.extend_from_slice(format!("col{}", c).as_bytes());
    }
    buf.push(b'\n');
    // Data rows
    for r in 0..num_rows {
        for c in 0..num_cols {
            if c > 0 {
                buf.push(b',');
            }
            let h = murmurhash3(&((r * num_cols + c) as u64).to_le_bytes());
            buf.extend_from_slice(format!("{}", h % 10000).as_bytes());
        }
        if r < num_rows - 1 {
            buf.push(b'\n');
        }
    }
    buf
}

// -- Kernel benchmarks ------------------------------------------------------

fn bench_tokenizer(input: &[u8], iters: usize) -> TimingSample {
    bench_iterations(
        || {
            // trim + split on space
            let trimmed = method_call(byte_string_expr(input), "trim_ascii", vec![]);
            let trimmed_val = eval_expr_val(&trimmed);
            match &trimmed_val {
                Value::ByteSlice(b) => {
                    let split = method_call(
                        byte_string_expr(b),
                        "split_byte",
                        vec![byte_char_expr(b' ')],
                    );
                    let _val = eval_expr_val(&split);
                }
                _ => {}
            }
        },
        iters,
    )
}

fn bench_vocab_counter(input: &[u8], iters: usize) -> TimingSample {
    bench_iterations(
        || {
            // Split tokens
            let split = method_call(
                byte_string_expr(input),
                "split_byte",
                vec![byte_char_expr(b' ')],
            );
            let tokens_val = eval_expr_val(&split);

            // Count (Rust side, simulating CJC program)
            let tokens = match &tokens_val {
                Value::Array(arr) => arr,
                _ => return,
            };

            let mut counts: std::collections::HashMap<u64, (Vec<u8>, i64)> =
                std::collections::HashMap::new();
            for t in tokens.iter() {
                match t {
                    Value::ByteSlice(b) => {
                        if !b.is_empty() {
                            let h = murmurhash3(b);
                            let entry = counts.entry(h).or_insert_with(|| ((**b).clone(), 0));
                            entry.1 += 1;
                        }
                    }
                    _ => {}
                }
            }

            // Sort deterministically
            let mut entries: Vec<(Vec<u8>, i64)> = counts.into_values().collect();
            entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            // Hash the output for determinism check
            let _output_hash: u64 = entries.iter().fold(0u64, |acc, (tok, count)| {
                acc ^ murmurhash3(tok) ^ murmurhash3(&count.to_le_bytes())
            });
        },
        iters,
    )
}

fn bench_csv_scanner(input: &[u8], iters: usize) -> TimingSample {
    bench_iterations(
        || {
            // Split rows on newline
            let rows = method_call(
                byte_string_expr(input),
                "split_byte",
                vec![byte_char_expr(b'\n')],
            );
            let rows_val = eval_expr_val(&rows);

            match &rows_val {
                Value::Array(row_arr) => {
                    // For each row, split on comma and count fields
                    for row in row_arr.iter() {
                        match row {
                            Value::ByteSlice(b) => {
                                let fields = method_call(
                                    byte_string_expr(b),
                                    "split_byte",
                                    vec![byte_char_expr(b',')],
                                );
                                let _fields_val = eval_expr_val(&fields);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        },
        iters,
    )
}

fn bench_determinism_gate(input: &[u8]) -> bool {
    // Run tokenize+hash twice, verify identical output
    let split1 = method_call(
        byte_string_expr(input),
        "split_byte",
        vec![byte_char_expr(b' ')],
    );
    let val1 = eval_expr_val(&split1);
    let hash1: u64 = match &val1 {
        Value::Array(tokens) => tokens.iter().fold(0u64, |acc, t| acc ^ value_hash(t)),
        _ => 0,
    };

    let split2 = method_call(
        byte_string_expr(input),
        "split_byte",
        vec![byte_char_expr(b' ')],
    );
    let val2 = eval_expr_val(&split2);
    let hash2: u64 = match &val2 {
        Value::Array(tokens) => tokens.iter().fold(0u64, |acc, t| acc ^ value_hash(t)),
        _ => 0,
    };

    hash1 == hash2 && hash1 != 0
}

// -- Main -------------------------------------------------------------------

fn main() {
    eprintln!("=== CJC NLP/ETL PREPROCESS BENCHMARK SUITE ===");
    eprintln!();

    // Configuration: (label, word_count/row_count, iterations)
    let text_sizes: Vec<(&str, usize, usize)> = vec![
        ("small", 100, 200),
        ("medium", 1_000, 50),
        ("large", 10_000, 10),
        ("xlarge", 50_000, 3),
    ];

    let csv_sizes: Vec<(&str, usize, usize, usize)> = vec![
        ("small", 100, 5, 200),
        ("medium", 1_000, 10, 50),
        ("large", 10_000, 10, 10),
        ("xlarge", 50_000, 5, 3),
    ];

    // ── Kernel A: Tokenizer ──
    eprintln!("--- Kernel A: Tokenizer (trim + split) ---");
    for (label, num_words, iters) in &text_sizes {
        let input = generate_text(*num_words);
        let sample = bench_tokenizer(&input, *iters);
        print_jsonl(&format!("tokenizer_{}", label), input.len(), &sample);
        print_scorecard_line(&format!("tokenizer_{}", label), input.len(), &sample);
    }
    eprintln!();

    // ── Kernel B: Vocab Counter ──
    eprintln!("--- Kernel B: Vocab Counter (split + count + sort) ---");
    for (label, num_words, iters) in &text_sizes {
        let input = generate_text(*num_words);
        let sample = bench_vocab_counter(&input, *iters);
        print_jsonl(&format!("vocab_{}", label), input.len(), &sample);
        print_scorecard_line(&format!("vocab_{}", label), input.len(), &sample);
    }
    eprintln!();

    // ── Kernel C: CSV Scanner ──
    eprintln!("--- Kernel C: CSV Scanner (row split + field split) ---");
    for (label, num_rows, num_cols, iters) in &csv_sizes {
        let input = generate_csv(*num_rows, *num_cols);
        let sample = bench_csv_scanner(&input, *iters);
        print_jsonl(&format!("csv_{}", label), input.len(), &sample);
        print_scorecard_line(&format!("csv_{}", label), input.len(), &sample);
    }
    eprintln!();

    // ── Determinism Gate ──
    eprintln!("--- Determinism Double-Run Gate ---");
    let mut all_deterministic = true;
    for (label, num_words, _) in &text_sizes {
        let input = generate_text(*num_words);
        let pass = bench_determinism_gate(&input);
        let status = if pass { "PASS" } else { "FAIL" };
        eprintln!("  det_gate_{:10} {}", label, status);
        println!(
            r#"{{"kernel":"det_gate_{}","input_bytes":{},"deterministic":{}}}"#,
            label,
            input.len(),
            pass
        );
        if !pass {
            all_deterministic = false;
        }
    }
    eprintln!();

    // ── Summary ──
    if all_deterministic {
        eprintln!("=== ALL BENCHMARKS COMPLETE — DETERMINISM GATES PASSED ===");
    } else {
        eprintln!("=== WARNING: DETERMINISM GATE FAILURES DETECTED ===");
        std::process::exit(1);
    }
}
