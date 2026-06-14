//! Executor parity tests for builtins.
//!
//! Validates that builtins produce identical results in both
//! cjc-eval (AST interpreter) and cjc-mir-exec (MIR executor).

/// Run CJC source through eval, return output lines.
fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    let _ = interp.exec(&program);
    interp.output
}

/// Run CJC source through MIR-exec, return output lines.
fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Assert both executors produce identical output.
fn assert_parity(src: &str) {
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "Parity mismatch!\nEval: {eval_out:?}\nMIR:  {mir_out:?}");
}

// ── Math builtins ──────────────────────────────────────────────

#[test]
fn parity_mean() {
    assert_parity(r#"
        let arr: Any = [1.0, 2.0, 3.0, 4.0, 5.0];
        print(mean(arr));
    "#);
}

#[test]
fn parity_erf() {
    assert_parity(r#"
        print(erf(1.0));
    "#);
}

#[test]
fn parity_erfc() {
    assert_parity(r#"
        print(erfc(1.0));
    "#);
}

// ── Sorting / selection ────────────────────────────────────────

#[test]
fn parity_sort() {
    assert_parity(r#"
        let arr: Any = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        print(sort(arr));
    "#);
}

#[test]
fn parity_abs_int_float() {
    assert_parity(r#"
        print(abs(-42));
        print(int(3.7));
        print(float(5));
    "#);
}

#[test]
fn parity_sqrt_floor_isnan() {
    assert_parity(r#"
        print(sqrt(16.0));
        print(floor(3.7));
        print(isnan(0.0));
    "#);
}

// ── String builtins ────────────────────────────────────────────

#[test]
fn parity_string_ops() {
    assert_parity(r#"
        print(str_to_upper("hello"));
        print(str_to_lower("WORLD"));
        print(str_trim("  spaced  "));
        print(str_starts("hello world", "hello"));
        print(str_ends("hello world", "world"));
    "#);
}

#[test]
fn parity_str_detect_count() {
    assert_parity(r#"
        print(str_detect("hello world", "world"));
        print(str_count("banana", "a"));
        print(str_sub("hello", 0, 3));
    "#);
}

// ── Tidy builtins ──────────────────────────────────────────────

#[test]
fn parity_col_desc_asc() {
    assert_parity(r#"
        let c: Any = col("x");
        print(c);
        let d: Any = desc("x");
        print(d);
        let a: Any = asc("x");
        print(a);
    "#);
}

// ── Closures bound to locals & higher-order calls ──────────────
//
// Regression guard for the AST-eval ↔ MIR-exec dispatch gap where
// calling a closure/Fn held in a local variable — `let f = |x| ..;
// f(x)` — errored `undefined function f` in cjc-eval while MIR-exec
// dispatched it correctly. See cjc-eval::dispatch_call Closure/Fn
// branch (mirrors cjc-mir-exec::dispatch_call). Discovered 2026-06-13,
// docs/T0_STAGE5B_ALLOC_ELISION.md §6.

#[test]
fn parity_closure_local_call_in_loop() {
    // The canonical repro: closure captures `offset`, called in a loop.
    assert_parity(r#"
        fn main() -> i64 {
            let offset: i64 = 100;
            let f = |x: i64| x + offset;
            let mut sum: i64 = 0;
            let mut i: i64 = 0;
            while i < 5 {
                sum = sum + f(i);
                i = i + 1;
            }
            return sum;
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_no_capture() {
    // A lambda with no free variables, bound to a local and called.
    assert_parity(r#"
        fn main() -> i64 {
            let double = |x: i64| x * 2;
            return double(21);
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_called_multiple_times() {
    // Same closure invoked repeatedly must agree on every call.
    assert_parity(r#"
        fn main() -> i64 {
            let k: i64 = 3;
            let scale = |x: i64| x * k;
            print(scale(1));
            print(scale(2));
            print(scale(10));
            return scale(0);
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_passed_as_argument() {
    // Higher-order: a closure flows into a param and is called there.
    assert_parity(r#"
        fn apply(g: Any, x: i64) -> i64 {
            return g(x);
        }
        fn main() -> i64 {
            let base: i64 = 10;
            let f = |y: i64| y + base;
            return apply(f, 5);
        }
        print(main());
    "#);
}

#[test]
fn parity_named_fn_value_passed_as_argument() {
    // Higher-order with a named function used as a first-class value
    // (`Value::Fn`) bound to a local, then dispatched.
    assert_parity(r#"
        fn inc(x: i64) -> i64 {
            return x + 1;
        }
        fn apply(g: Any, x: i64) -> i64 {
            return g(x);
        }
        fn main() -> i64 {
            let h = inc;
            return apply(h, 41);
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_returned_value_used() {
    // Two closures in scope; selecting and calling via locals.
    assert_parity(r#"
        fn main() -> i64 {
            let a: i64 = 7;
            let b: i64 = 100;
            let addA = |x: i64| x + a;
            let addB = |x: i64| x + b;
            return addA(1) + addB(1);
        }
        print(main());
    "#);
}

// ── Escaping closures: lexical capture parity ──────────────────
//
// A closure that outlives its defining scope must observe the values
// captured at creation time (lexical), identically in both executors.
// Before cjc-eval grew real capture it either errored (`undefined
// variable`) or read the caller's live scope (dynamic scoping); these
// pin the fixed behavior to MIR-exec's lexical capture.

#[test]
fn parity_closure_escapes_factory() {
    // The captured `n` is absent from the caller's scope — only lexical
    // capture can resolve it. Expected: 8 (3 + captured 5).
    assert_parity(r#"
        fn make_adder(n: i64) -> Any {
            let f = |x: i64| x + n;
            return f;
        }
        fn main() -> i64 {
            let add5 = make_adder(5);
            return add5(3);
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_escapes_with_shadow() {
    // The caller rebinds `n` after the closure is built. Lexical capture
    // must ignore the caller's `n` and use the captured 5 → 8.
    assert_parity(r#"
        fn make_adder(n: i64) -> Any {
            let f = |x: i64| x + n;
            return f;
        }
        fn main() -> i64 {
            let add5 = make_adder(5);
            let n: i64 = 1000;
            return add5(3) + n - n;
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_factory_multiple_instances() {
    // Two closures from the same factory capture distinct values.
    assert_parity(r#"
        fn make_adder(n: i64) -> Any {
            let f = |x: i64| x + n;
            return f;
        }
        fn main() -> i64 {
            let add5 = make_adder(5);
            let add10 = make_adder(10);
            print(add5(1));
            print(add10(1));
            return add5(0) + add10(0);
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_capture_is_snapshot() {
    // Mutating the captured variable after the closure is built must not
    // change what the closure sees (capture is by value at creation).
    assert_parity(r#"
        fn main() -> i64 {
            let mut n: i64 = 1;
            let f = |x: i64| x + n;
            n = 100;
            return f(0);
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_captures_multiple_vars() {
    // A closure capturing several distinct free variables.
    assert_parity(r#"
        fn main() -> i64 {
            let a: i64 = 3;
            let b: i64 = 40;
            let c: i64 = 500;
            let f = |x: i64| x + a + b + c;
            return f(6000);
        }
        print(main());
    "#);
}

#[test]
fn parity_closure_returned_then_called_in_loop() {
    // Factory-built closure invoked repeatedly after escaping.
    assert_parity(r#"
        fn make_scaler(k: i64) -> Any {
            return |x: i64| x * k;
        }
        fn main() -> i64 {
            let triple = make_scaler(3);
            let mut sum: i64 = 0;
            let mut i: i64 = 0;
            while i < 4 {
                sum = sum + triple(i);
                i = i + 1;
            }
            return sum;
        }
        print(main());
    "#);
}

#[test]
fn parity_capturing_closure_through_higher_order_builtins() {
    // A capturing closure passed to array_map / array_reduce exercises
    // eval's now-live Value::Closure arms in those builtins (env prepended
    // to the per-element args). Must match MIR-exec.
    assert_parity(r#"
        fn main() -> i64 {
            let bias: i64 = 10;
            let shift = |x: i64| x + bias;
            let arr: Any = [1, 2, 3];
            let mapped: Any = array_map(arr, shift);
            let total: i64 = array_reduce(arr, 0, |acc: i64, x: i64| acc + x + bias);
            print(mapped);
            print(total);
            return 0;
        }
        print(main());
    "#);
}

// ── Determinism: same seed = identical output ──────────────────

#[test]
fn determinism_both_executors_same_seed() {
    let src = r#"
        let x: Any = [5.0, 3.0, 1.0, 4.0, 2.0];
        print(sort(x));
        print(mean(x));
        print(sqrt(16.0));
        print(abs(-7));
    "#;
    let eval1 = run_eval(src, 123);
    let eval2 = run_eval(src, 123);
    let mir1 = run_mir(src, 123);
    let mir2 = run_mir(src, 123);
    assert_eq!(eval1, eval2, "Eval not deterministic");
    assert_eq!(mir1, mir2, "MIR not deterministic");
    assert_eq!(eval1, mir1, "Eval/MIR not identical");
}
