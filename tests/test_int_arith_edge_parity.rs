//! Integer-arithmetic edge cases must behave identically — and never panic —
//! in AST-eval, MIR-exec, and MIR-exec with the optimizer (constant folding).
//!
//! Regressions:
//! - `i64::MIN / -1` and `i64::MIN % -1` panicked ("attempt to divide with
//!   overflow") in both executors and in both constant folders. Semantics now
//!   wrap, matching `+ - *`: `MIN / -1 == MIN`, `MIN % -1 == 0`.
//! - The optimizer folded int `**` with an exact `wrapping_pow`, while both
//!   executors compute it through f64 (saturating). `--mir-opt` printed a
//!   different number for results beyond 2^53 (e.g. `3 ** 39`, `3 ** 40`).

fn run_eval(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in {src:?}");
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output
}

fn run_mir(src: &str, optimize: bool) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors in {src:?}");
    let r = if optimize {
        cjc_mir_exec::run_program_optimized_with_executor(&program, 42)
    } else {
        cjc_mir_exec::run_program_with_executor(&program, 42)
    };
    r.expect("mir failed").1.output
}

/// Asserts all three paths agree and returns the shared output.
fn run_all(src: &str) -> Vec<String> {
    let eval = run_eval(src);
    let mir = run_mir(src, false);
    let opt = run_mir(src, true);
    assert_eq!(eval, mir, "AST-eval vs MIR-exec differ for {src}");
    assert_eq!(mir, opt, "MIR-exec vs optimized MIR-exec differ for {src}");
    mir
}

const MIN: &str = "-9223372036854775808";

#[test]
fn min_div_neg_one_wraps_at_runtime() {
    let out = run_all(
        "let m: i64 = -9223372036854775807 - 1;
let n: i64 = -1;
print(m / n);
print(m % n);",
    );
    assert_eq!(out, vec![MIN, "0"]);
}

#[test]
fn min_div_neg_one_wraps_when_constant_folded() {
    let out = run_all(
        "print((-9223372036854775807 - 1) / -1);
print((-9223372036854775807 - 1) % -1);",
    );
    assert_eq!(out, vec![MIN, "0"]);
}

#[test]
fn ordinary_div_mod_unchanged() {
    let out = run_all("print(7 / 2); print(-7 / 2); print(7 % 3); print(-7 % 3);");
    // Truncating division, remainder takes the sign of the dividend.
    assert_eq!(out, vec!["3", "-3", "1", "-1"]);
}

#[test]
fn neg_min_wraps_in_every_path() {
    let out = run_all(
        "let m: i64 = -9223372036854775807 - 1;
print(-m);
print(-(-9223372036854775807 - 1));",
    );
    assert_eq!(out, vec![MIN, MIN]);
}

/// Regression (found by `fuzz_optimizer_parity`): strength reduction rewrote
/// `x * 0 => 0` for any `x`, erasing `x`'s runtime error under `--mir-opt`.
#[test]
fn mul_by_zero_keeps_operand_error() {
    for src in ["print((0 % 0) * 0);", "print(0 * (1 / 0));"] {
        let (program, _) = cjc_parser::parse_source(src);
        let unopt = cjc_mir_exec::run_program(&program, 42);
        let opt = cjc_mir_exec::run_program_optimized(&program, 42);
        assert!(unopt.is_err(), "{src}: expected a runtime error");
        assert!(opt.is_err(), "{src}: optimizer erased the runtime error");
    }
}

/// `t * 0` for a tensor is a zero tensor, not `Int 0`: strength reduction
/// must not drop an operand of unknown type.
#[test]
fn tensor_times_zero_stays_tensor() {
    let src = "let t = Tensor.from_vec([1.0, 2.0], [2]);
print(t * 0);";
    let out = run_all(src);
    assert_ne!(out, vec!["0"], "tensor * 0 was rewritten to Int 0");
}

#[test]
fn int_pow_folding_matches_runtime() {
    // Literal forms are folded by the optimizer; variable forms are not.
    let src = "let three: i64 = 3;
let two: i64 = 2;
print(3 ** 39);
print(three ** 39);
print(3 ** 40);
print(three ** 40);
print(2 ** 10);
print(two ** 10);";
    let out = run_all(src);
    assert_eq!(out[0], out[1], "folded vs runtime 3**39");
    assert_eq!(out[2], out[3], "folded vs runtime 3**40");
    assert_eq!(out[4], "1024");
    assert_eq!(out[5], "1024");
}
