//! Performance benchmarks for hardening council features.
//!
//! All benchmarks are #[ignore] // Perf benchmark — run with: cargo test h12_perf -- --ignored by default — run with:
//! `cargo test --test test_hardening h12_perf -- --ignored`

use std::time::Instant;

// ── TiledMatmul performance ─────────────────────────────────────────

#[test]
#[ignore] // Perf benchmark — run with: cargo test h12_perf -- --ignored
fn h12_perf_tiled_matmul_64x64() {
    use cjc_runtime::tensor::Tensor;
    let n = 64;
    let a_data: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.01).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64 * 0.01).collect();
    let a = Tensor::from_vec(a_data, &[n, n]).unwrap();
    let b = Tensor::from_vec(b_data, &[n, n]).unwrap();

    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        let _ = a.matmul(&b).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    eprintln!(
        "64x64 matmul (tiled): {:.2?} per iter ({} iters in {:.2?})",
        per_iter, iters, elapsed,
    );
}

#[test]
#[ignore] // Perf benchmark — run with: cargo test h12_perf -- --ignored
fn h12_perf_tiled_matmul_128x128() {
    use cjc_runtime::tensor::Tensor;
    let n = 128;
    let a_data: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.001).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64 * 0.001).collect();
    let a = Tensor::from_vec(a_data, &[n, n]).unwrap();
    let b = Tensor::from_vec(b_data, &[n, n]).unwrap();

    let start = Instant::now();
    let iters = 20;
    for _ in 0..iters {
        let _ = a.matmul(&b).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    eprintln!(
        "128x128 matmul (tiled): {:.2?} per iter ({} iters in {:.2?})",
        per_iter, iters, elapsed,
    );
}

// ── Window function performance ─────────────────────────────────────

#[test]
#[ignore] // Perf benchmark — run with: cargo test h12_perf -- --ignored
fn h12_perf_window_sum_10k() {
    use cjc_runtime::window;
    let data: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.1).collect();

    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        let _ = window::window_sum(&data, 100);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    eprintln!(
        "window_sum(10k, w=100): {:.2?} per iter ({} iters in {:.2?})",
        per_iter, iters, elapsed,
    );
}

// ── JSON performance ────────────────────────────────────────────────

#[test]
#[ignore] // Perf benchmark — run with: cargo test h12_perf -- --ignored
fn h12_perf_json_roundtrip() {
    use cjc_runtime::json;
    use cjc_runtime::Value;

    let input = r#"{"users":[{"name":"Alice","age":30,"active":true},{"name":"Bob","age":25,"active":false}],"count":2,"metadata":{"version":"1.0","tags":["a","b","c"]}}"#;

    let start = Instant::now();
    let iters = 1000;
    for _ in 0..iters {
        let val = json::json_parse(input).unwrap();
        let _ = json::json_stringify(&val).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    eprintln!(
        "JSON parse+stringify roundtrip: {:.2?} per iter ({} iters in {:.2?})",
        per_iter, iters, elapsed,
    );
}

// ── DateTime performance ────────────────────────────────────────────

#[test]
#[ignore] // Perf benchmark — run with: cargo test h12_perf -- --ignored
fn h12_perf_datetime_operations() {
    use cjc_runtime::datetime;

    let start = Instant::now();
    let iters = 10_000;
    for i in 0..iters {
        let dt = datetime::datetime_from_parts(2024, 1, 1, 0, 0, 0);
        let _ = datetime::datetime_year(dt);
        let _ = datetime::datetime_month(dt);
        let _ = datetime::datetime_day(dt);
        let _ = datetime::datetime_format(dt);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    eprintln!(
        "DateTime from_parts+extract+format: {:.2?} per iter ({} iters in {:.2?})",
        per_iter, iters, elapsed,
    );
}

// ── Module system performance ───────────────────────────────────────

#[test]
#[ignore] // Perf benchmark — run with: cargo test h12_perf -- --ignored
fn h12_perf_module_graph_build() {
    use std::fs;

    let dir = tempfile::tempdir().unwrap();
    // Create 10 modules with imports
    let mut main_imports = String::new();
    for i in 0..10 {
        let name = format!("mod{}.cjc", i);
        let content = format!("fn mod{}_fn() -> i64 {{ {} }}", i, i);
        fs::write(dir.path().join(&name), content).unwrap();
        main_imports.push_str(&format!("import mod{}\n", i));
    }
    main_imports.push_str("let x = 1;");
    fs::write(dir.path().join("main.cjc"), &main_imports).unwrap();

    let entry = dir.path().join("main.cjc");
    let start = Instant::now();
    let iters = 50;
    for _ in 0..iters {
        let graph = cjc_module::build_module_graph(&entry).unwrap();
        let _ = cjc_module::merge_programs(&graph).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    eprintln!(
        "Module graph build+merge (10 modules): {:.2?} per iter ({} iters in {:.2?})",
        per_iter, iters, elapsed,
    );
}
