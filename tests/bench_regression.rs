//! Performance regression smoke tests.
//!
//! These are lightweight timing checks that catch order-of-magnitude regressions.
//! They don't enforce specific thresholds — just verify key operations complete
//! within a generous wall-clock budget (10x expected).
//!
//! Run with: cargo test --test bench_regression -- --ignored
//! Or in CI: include in the perf gate job.

use std::time::Instant;

/// Matmul 64x64 should complete in < 100ms (typically ~1ms).
#[test]
#[ignore] // Perf regression gate — run with: cargo test --test bench_regression -- --ignored
fn bench_matmul_64x64_regression() {
    use cjc_runtime::tensor::Tensor;
    let n = 64;
    let data: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.001).collect();
    let a = Tensor::from_vec(data.clone(), &[n, n]).unwrap();
    let b = Tensor::from_vec(data, &[n, n]).unwrap();

    let start = Instant::now();
    for _ in 0..10 {
        let _ = a.matmul(&b).unwrap();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 5000,
        "matmul 64x64 x10 took {}ms — possible regression",
        elapsed.as_millis()
    );
}

/// Parse + eval a 100-line program should complete in < 2s.
#[test]
#[ignore] // Perf regression gate — run with: cargo test --test bench_regression -- --ignored
fn bench_parse_eval_regression() {
    let mut src = String::new();
    src.push_str("let total = 0.0\n");
    for i in 0..100 {
        src.push_str(&format!("total = total + {}.0 * 0.001\n", i));
    }
    src.push_str("total\n");

    let start = Instant::now();
    for _ in 0..50 {
        let (program, diags) = cjc_parser::parse_source(&src);
        assert!(!diags.has_errors());
        let mut interp = cjc_eval::Interpreter::new(42);
        let _ = interp.exec(&program);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 10000,
        "parse+eval 100-line x50 took {}ms — possible regression",
        elapsed.as_millis()
    );
}

/// Cholesky 32x32 should complete in < 500ms.
#[test]
#[ignore] // Perf regression gate — run with: cargo test --test bench_regression -- --ignored
fn bench_cholesky_regression() {
    use cjc_runtime::tensor::Tensor;
    let n = 32;
    // Create SPD matrix: A = I + X^T * X
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 10.0; // dominant diagonal ensures SPD
        for j in 0..n {
            data[i * n + j] += (i as f64 * 0.1) * (j as f64 * 0.1);
        }
    }
    let mat = Tensor::from_vec(data, &[n, n]).unwrap();

    let start = Instant::now();
    for _ in 0..100 {
        let _ = mat.cholesky().unwrap();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 5000,
        "cholesky 32x32 x100 took {}ms — possible regression",
        elapsed.as_millis()
    );
}

/// Kahan accumulator: sum 1M values should be < 500ms.
#[test]
#[ignore] // Perf regression gate — run with: cargo test --test bench_regression -- --ignored
fn bench_kahan_sum_regression() {
    use cjc_repro::KahanAccumulatorF64;

    let start = Instant::now();
    for _ in 0..10 {
        let mut acc = KahanAccumulatorF64::new();
        for i in 0..1_000_000 {
            acc.add(i as f64 * 1e-9);
        }
        let _ = acc.finalize();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 5000,
        "kahan 1M x10 took {}ms — possible regression",
        elapsed.as_millis()
    );
}

/// MIR-exec pipeline: parse + lower + execute a simple program.
#[test]
#[ignore] // Perf regression gate — run with: cargo test --test bench_regression -- --ignored
fn bench_mir_exec_regression() {
    let src = r#"
fn fib(n: i64) -> i64 {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}
let result = fib(20)
result
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let start = Instant::now();
    for _ in 0..5 {
        let _ = cjc_mir_exec::run_program(&program, 42);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 30000,
        "MIR-exec fib(20) x5 took {}ms — possible regression",
        elapsed.as_millis()
    );
}
