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
