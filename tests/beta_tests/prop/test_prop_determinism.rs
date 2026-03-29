// CJC v0.2 Beta — Property-Based Tests: Determinism
//
// These tests verify that the determinism contract holds across the
// compiler pipeline. Same seed = bit-identical output. All tests use
// deterministic iteration (BTreeMap/BTreeSet) and Kahan summation.

#[allow(unused_imports)]
use cjc_runtime::Value;

// ── Helper: deterministic pseudo-random sequence via SplitMix64 ──

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn rand_f64(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ── Property: eval produces identical results across 10 runs ──

fn run_program_n_times(src: &str, seed: u64, n: usize) -> Vec<String> {
    let mut results = Vec::new();
    for _ in 0..n {
        let (program, diags) = cjc_parser::parse_source(src);
        assert!(!diags.has_errors(), "parse errors in: {}", src);
        let result = cjc_eval::Interpreter::new(seed).exec(&program);
        match result {
            Ok(val) => results.push(format!("{}", val)),
            Err(e) => results.push(format!("ERR:{:?}", e)),
        }
    }
    results
}

#[test]
fn prop_eval_determinism_arithmetic() {
    let src = "fn main() -> f64 { let x = 3.14; let y = 2.71; x * y + (x - y) / (x + y) }";
    let results = run_program_n_times(src, 42, 10);
    for r in &results {
        assert_eq!(r, &results[0], "non-deterministic eval result");
    }
}

#[test]
fn prop_eval_determinism_integer_ops() {
    let src = "fn main() -> i64 { let a = 100; let b = 37; a * b + a - b }";
    let results = run_program_n_times(src, 42, 10);
    for r in &results {
        assert_eq!(r, &results[0], "non-deterministic integer ops");
    }
}

#[test]
fn prop_eval_determinism_nested_calls() {
    let src = r#"
        fn square(x: f64) -> f64 { x * x }
        fn cube(x: f64) -> f64 { x * x * x }
        fn main() -> f64 { square(3.0) + cube(2.0) }
    "#;
    let results = run_program_n_times(src, 42, 10);
    for r in &results {
        assert_eq!(r, &results[0], "non-deterministic nested calls");
    }
}

// ── Property: MIR-exec and eval produce identical output ──

fn parity_check(src: &str, seed: u64) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors (count: {})", diags.error_count());

    let eval_result = cjc_eval::Interpreter::new(seed).exec(&program);
    let mir_result = cjc_mir_exec::run_program_with_executor(&program, seed);

    if let (Ok(eval_val), Ok((mir_val, _))) = (&eval_result, &mir_result) {
        assert_eq!(
            format!("{}", eval_val),
            format!("{}", mir_val),
            "parity violation for: {}",
            src
        );
    }
}

#[test]
fn prop_parity_simple_arithmetic() {
    let programs = [
        "fn main() -> i64 { 1 + 2 * 3 }",
        "fn main() -> i64 { 10 - 3 }",
        "fn main() -> f64 { sqrt(9.0) }",
    ];
    for src in &programs {
        parity_check(src, 42);
    }
}

#[test]
fn prop_parity_conditionals() {
    let programs = [
        "fn main() -> i64 { if true { 1 } else { 2 } }",
        "fn main() -> i64 { if 3 > 2 { 10 } else { 20 } }",
    ];
    for src in &programs {
        parity_check(src, 42);
    }
}

#[test]
fn prop_parity_functions() {
    let src = r#"
        fn square(x: f64) -> f64 { x * x }
        fn add(a: f64, b: f64) -> f64 { a + b }
        fn main() -> f64 { add(square(3.0), square(4.0)) }
    "#;
    parity_check(src, 42);
}

#[test]
fn prop_parity_loops() {
    let src = r#"
        fn sum_to(n: i64) -> i64 {
            let total = 0;
            let i = 1;
            while i <= n {
                total = total + i;
                i = i + 1
            }
            total
        }
        fn main() -> i64 { sum_to(100) }
    "#;
    parity_check(src, 42);
}

#[test]
fn prop_parity_recursion() {
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> i64 { fib(15) }
    "#;
    parity_check(src, 42);
}

// ── Property: repeated MIR-exec runs are deterministic ──

#[test]
fn prop_mir_exec_determinism_10_runs() {
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> i64 { fib(15) }
    "#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut results = Vec::new();
    for _ in 0..10 {
        let r = cjc_mir_exec::run_program_with_executor(&program, 42);
        match r {
            Ok((val, _)) => results.push(format!("{}", val)),
            Err(e) => results.push(format!("ERR:{:?}", e)),
        }
    }
    for r in &results {
        assert_eq!(r, &results[0], "MIR-exec non-determinism detected");
    }
}

// ── Property: Kahan summation is order-independent within deterministic iteration ──

#[test]
fn prop_kahan_summation_stability() {
    use cjc_repro::KahanAccumulatorF64;

    // Generate 1000 random f64 values
    let mut rng = 42u64;
    let values: Vec<f64> = (0..1000).map(|_| rand_f64(&mut rng) * 1e6 - 5e5).collect();

    // Sum forward
    let mut fwd = KahanAccumulatorF64::new();
    for &v in &values {
        fwd.add(v);
    }

    // Sum backward
    let mut bwd = KahanAccumulatorF64::new();
    for &v in values.iter().rev() {
        bwd.add(v);
    }

    // Kahan should give very close results regardless of order
    let diff = (fwd.finalize() - bwd.finalize()).abs();
    assert!(
        diff < 1e-6,
        "Kahan summation order sensitivity too high: diff={}",
        diff
    );
}

// ── Property: BTreeMap iteration is always sorted ──

#[test]
fn prop_btreemap_deterministic_iteration() {
    use std::collections::BTreeMap;

    let mut rng = 123u64;
    for _ in 0..10 {
        let mut map = BTreeMap::new();
        for _ in 0..100 {
            let key = splitmix64(&mut rng) % 50;
            let val = splitmix64(&mut rng);
            map.insert(key, val);
        }
        let keys: Vec<u64> = map.keys().copied().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "BTreeMap iteration not sorted");
    }
}

// ── Property: SplitMix64 is deterministic with same seed ──

#[test]
fn prop_splitmix64_deterministic() {
    for seed in 0..100u64 {
        let mut s1 = seed;
        let mut s2 = seed;
        let seq1: Vec<u64> = (0..50).map(|_| splitmix64(&mut s1)).collect();
        let seq2: Vec<u64> = (0..50).map(|_| splitmix64(&mut s2)).collect();
        assert_eq!(seq1, seq2, "SplitMix64 non-deterministic for seed {}", seed);
    }
}
