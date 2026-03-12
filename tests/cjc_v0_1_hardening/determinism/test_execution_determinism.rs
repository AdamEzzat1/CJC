//! Execution determinism: same program + same seed = identical output (bit-exact).

#[path = "../helpers.rs"]
mod helpers;
use helpers::*;

/// Helper: run the same program N times and verify all outputs are identical.
fn assert_deterministic(label: &str, src: &str, runs: usize) {
    let first = run_mir(src);
    for i in 1..runs {
        let current = run_mir(src);
        assert_eq!(
            first, current,
            "[DETERMINISM FAILURE: {label}, run {i}]\nFirst: {first:?}\nCurrent: {current:?}"
        );
    }
}

/// Helper: run with multiple seeds and verify different outputs per seed.
fn assert_seed_sensitivity(src: &str, seeds: &[u64]) {
    let mut outputs: Vec<Vec<String>> = Vec::new();
    for &seed in seeds {
        outputs.push(run_mir_seeded(src, seed));
    }
    // At least some outputs should differ
    let all_same = outputs.windows(2).all(|w| w[0] == w[1]);
    assert!(
        !all_same || outputs[0].is_empty(),
        "Different seeds should produce different outputs for programs using RNG"
    );
}

// ============================================================
// Basic determinism
// ============================================================

#[test]
fn det_integer_arithmetic() {
    assert_deterministic("int arith", "fn main() { print(1 + 2 * 3 - 4); }", 5);
}

#[test]
fn det_float_arithmetic() {
    assert_deterministic("float arith", "fn main() { print(1.5 + 2.5 * 3.0); }", 5);
}

#[test]
fn det_function_calls() {
    assert_deterministic("fn calls", r#"
fn f(x: i64) -> i64 { x * 2 + 1 }
fn g(x: i64) -> i64 { f(x) + f(x + 1) }
fn main() { print(g(10)); }
"#, 5);
}

#[test]
fn det_recursive_fibonacci() {
    assert_deterministic("fib", r#"
fn fib(n: i64) -> i64 {
    if n <= 1 { return n; }
    fib(n - 1) + fib(n - 2)
}
fn main() { print(fib(15)); }
"#, 3);
}

#[test]
fn det_loop_accumulation() {
    assert_deterministic("loop acc", r#"
fn main() {
    let mut sum: i64 = 0;
    for i in 0..100 {
        sum = sum + i;
    }
    print(sum);
}
"#, 5);
}

#[test]
fn det_nested_loops() {
    assert_deterministic("nested loops", r#"
fn main() {
    let mut total: i64 = 0;
    for i in 0..10 {
        for j in 0..10 {
            total = total + i * j;
        }
    }
    print(total);
}
"#, 3);
}

#[test]
fn det_closure_execution() {
    assert_deterministic("closure", r#"
fn main() {
    let n: i64 = 10;
    let f = |x: i64| x + n;
    let mut sum: i64 = 0;
    for i in 0..20 {
        sum = sum + f(i);
    }
    print(sum);
}
"#, 5);
}

#[test]
fn det_match_expression() {
    assert_deterministic("match", r#"
fn classify(x: i64) -> i64 {
    match x {
        0 => 100,
        1 => 200,
        2 => 300,
        _ => 0,
    }
}
fn main() {
    let mut sum: i64 = 0;
    for i in 0..5 {
        sum = sum + classify(i);
    }
    print(sum);
}
"#, 3);
}

#[test]
fn det_struct_operations() {
    assert_deterministic("struct", r#"
struct Vec2 { x: f64, y: f64 }
fn mag(v: Vec2) -> f64 {
    sqrt(v.x * v.x + v.y * v.y)
}
fn main() {
    let v = Vec2 { x: 3.0, y: 4.0 };
    print(mag(v));
}
"#, 5);
}

// ============================================================
// RNG-dependent determinism
// ============================================================

#[test]
fn det_rng_same_seed() {
    let src = r#"
fn main() {
    let t = Tensor.randn([5]);
    print(t);
}
"#;
    // Same seed → same random tensor
    let out1 = run_mir_seeded(src, 42);
    let out2 = run_mir_seeded(src, 42);
    assert_eq!(out1, out2, "Same seed should produce identical random tensors");
}

#[test]
fn det_rng_different_seeds() {
    let src = r#"
fn main() {
    let t = Tensor.randn([10]);
    print(t);
}
"#;
    assert_seed_sensitivity(src, &[1, 2, 3]);
}

// ============================================================
// Multi-run determinism (critical for reproducible computation)
// ============================================================

#[test]
fn det_numerical_accumulation() {
    assert_deterministic("accumulation", r#"
fn main() {
    let t = Tensor.from_vec([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [10]);
    print(t.sum());
}
"#, 10);
}

#[test]
fn det_matmul() {
    assert_deterministic("matmul", r#"
fn main() {
    let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [4]);
    let a = a.reshape([2, 2]);
    let b = Tensor.from_vec([5.0, 6.0, 7.0, 8.0], [4]);
    let b = b.reshape([2, 2]);
    let c = matmul(a, b);
    print(c);
}
"#, 5);
}

#[test]
fn det_builtin_math() {
    assert_deterministic("math builtins", r#"
fn main() {
    print(sin(1.0));
    print(cos(1.0));
    print(exp(1.0));
    print(log(2.0));
    print(sqrt(2.0));
    print(pow(2.0, 0.5));
}
"#, 5);
}

// ============================================================
// Eval vs MIR determinism (cross-executor)
// ============================================================

#[test]
fn det_cross_executor_parity() {
    let programs = vec![
        "fn main() { print(42); }",
        "fn main() { print(1 + 2 * 3); }",
        r#"fn main() { print("hello"); }"#,
        "fn main() { let x: i64 = 10; print(x * x); }",
        r#"
fn f(x: i64) -> i64 { x + 1 }
fn main() { print(f(41)); }
"#,
    ];
    for (i, src) in programs.iter().enumerate() {
        let eval = run_eval(src);
        let mir = run_mir(src);
        assert_eq!(eval, mir, "Cross-executor parity failure on program {i}");
    }
}
