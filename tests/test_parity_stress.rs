// ═══════════════════════════════════════════════════════════════════════
// Parity Stress Tests
//
// Generates programs from templates and verifies parity across 50 seeds.
// For each seed, both executors must produce bit-identical output.
// ═══════════════════════════════════════════════════════════════════════

/// Run CJC source through eval with given seed, return output lines.
fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Parse errors: {:?}", diags.render_all(src, "<parity-stress>"));
    let mut interp = cjc_eval::Interpreter::new(seed);
    let _ = interp.exec(&program);
    interp.output
}

/// Run CJC source through MIR-exec with given seed, return output lines.
fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Parse errors: {:?}", diags.render_all(src, "<parity-stress>"));
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed (seed={seed}): {e}"));
    executor.output
}

/// Assert parity at a specific seed.
fn assert_parity_at_seed(src: &str, seed: u64) {
    let eval_out = run_eval(src, seed);
    let mir_out = run_mir(src, seed);
    assert_eq!(
        eval_out, mir_out,
        "Parity violation at seed={seed}!\nEval: {eval_out:?}\nMIR:  {mir_out:?}"
    );
}

/// Assert parity across seeds 1..=50.
fn assert_parity_all_seeds(src: &str) {
    for seed in 1..=50 {
        assert_parity_at_seed(src, seed);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Template 1: Arithmetic expressions
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_arithmetic() {
    assert_parity_all_seeds(r#"
        fn main() {
            let a: Any = 17;
            let b: Any = 5;
            print(a + b);
            print(a - b);
            print(a * b);
            print(a / b);
            print(a % b);
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 2: Function calls
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_function_calls() {
    assert_parity_all_seeds(r#"
        fn add(x: i64, y: i64) -> i64 { x + y }
        fn mul(x: i64, y: i64) -> i64 { x * y }
        fn main() {
            print(add(3, 4));
            print(mul(5, 6));
            print(add(mul(2, 3), 4));
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 3: Tensor operations (deterministic, no RNG)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_tensor_ops() {
    assert_parity_all_seeds(r#"
        fn main() {
            let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
            let b = Tensor.from_vec([5.0, 6.0, 7.0, 8.0], [2, 2]);
            let c = a + b;
            print(c);
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 4: Tensor.randn (seed-dependent, must be deterministic)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_tensor_randn_determinism() {
    // Tensor.randn uses the executor seed -- same seed must give same output
    let src = r#"
        fn main() {
            let t = Tensor.randn([2, 3]);
            print(t);
        }
    "#;
    // Run at each seed and verify both executors agree
    for seed in 1..=50 {
        assert_parity_at_seed(src, seed);
    }
    // Also verify determinism: same seed = same output
    let eval1 = run_eval(src, 7);
    let eval2 = run_eval(src, 7);
    assert_eq!(eval1, eval2, "Eval not deterministic at seed=7");
    let mir1 = run_mir(src, 7);
    let mir2 = run_mir(src, 7);
    assert_eq!(mir1, mir2, "MIR not deterministic at seed=7");
}

// ═══════════════════════════════════════════════════════════════════════
// Template 5: Array manipulations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_array_ops() {
    assert_parity_all_seeds(r#"
        fn main() {
            let mut arr: Any = [10, 20, 30];
            arr = array_push(arr, 40);
            arr = array_push(arr, 50);
            print(len(arr));
            print(arr[0]);
            print(arr[4]);
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 6: String operations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_string_ops() {
    assert_parity_all_seeds(r#"
        fn main() {
            let s: Any = "hello";
            print(len(s));
            print(str_to_upper(s));
            print(s + " world");
            print(to_string(42));
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 7: If/else branches
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_if_else() {
    assert_parity_all_seeds(r#"
        fn classify(x: i64) -> i64 {
            if x > 10 {
                1
            } else {
                if x > 5 {
                    2
                } else {
                    3
                }
            }
        }
        fn main() {
            print(classify(15));
            print(classify(7));
            print(classify(2));
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 8: While loops
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_while_loop() {
    assert_parity_all_seeds(r#"
        fn main() {
            let mut i: Any = 0;
            let mut product: Any = 1;
            while i < 6 {
                product = product * (i + 1);
                i = i + 1;
            }
            print(product);
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 9: Combined -- function + loop + array
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_combined_program() {
    assert_parity_all_seeds(r#"
        fn sum_to(n: i64) -> i64 {
            let mut s: Any = 0;
            let mut i: Any = 1;
            while i <= n {
                s = s + i;
                i = i + 1;
            }
            s
        }
        fn main() {
            print(sum_to(10));
            print(sum_to(100));
            let arr: Any = [sum_to(5), sum_to(10), sum_to(15)];
            print(arr);
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 10: Nested for-loops
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_nested_for_loops() {
    assert_parity_all_seeds(r#"
        fn main() {
            let mut total: Any = 0;
            for i in 0..5 {
                for j in 0..5 {
                    total = total + i * j;
                }
            }
            print(total);
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Template 11: Floating-point math builtins
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stress_float_builtins() {
    assert_parity_all_seeds(r#"
        fn main() {
            print(sqrt(144.0));
            print(abs(-99.5));
            print(floor(7.9));
            print(ceil(7.1));
        }
    "#);
}
