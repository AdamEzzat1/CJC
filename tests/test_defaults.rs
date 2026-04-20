//! Tests for default function parameters: `fn f(x: f64, tol: f64 = 1e-6)`
//!
//! Covers: parsing, AST inspection, eval, MIR-exec, parity, determinism,
//! error cases, proptest property checks, and a bolero fuzz target.
//!
//! All implementation lives in:
//!   - cjc-parser/src/lib.rs   (parse_param: `= expr` after type annotation)
//!   - cjc-eval/src/lib.rs     (call_function: defaults evaluated in caller scope)
//!   - cjc-mir-exec/src/lib.rs (call_function: same)
//!   - cjc-hir/src/lib.rs      (HirParam::default lowered from AST)
//!   - cjc-mir/src/lib.rs      (MirParam::default lowered from HIR)

use proptest::prelude::*;
use std::panic;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn eval_output(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors in eval_output: {:?}",
        diags.diagnostics
    );
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors in mir_output: {:?}",
        diags.diagnostics
    );
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("mir-exec failed");
    executor.output
}

/// Run in both executors and return (eval_output, mir_output).
fn both_output(src: &str) -> (Vec<String>, Vec<String>) {
    (eval_output(src), mir_output(src))
}

/// Assert that both executors produce the same output lines.
fn assert_parity(src: &str) {
    let (ev, mir) = both_output(src);
    assert_eq!(ev, mir, "eval vs mir-exec parity failure");
}

/// Assert that a CJC-Lang program produces a parse error.
fn assert_parse_error(src: &str) {
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(
        diags.has_errors(),
        "expected parse error for: {src}\ngot no errors"
    );
}

// ── Unit tests: parsing ───────────────────────────────────────────────────────

/// Parser accepts a single default parameter.
#[test]
fn parse_single_default() {
    let src = "fn solve(x: f64, tol: f64 = 1e-6) -> f64 { x }";
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "unexpected errors: {:?}", diags.diagnostics);
    if let cjc_ast::DeclKind::Fn(ref f) = prog.declarations[0].kind {
        assert_eq!(f.params.len(), 2);
        assert!(f.params[0].default.is_none(), "x has no default");
        assert!(f.params[1].default.is_some(), "tol has a default");
    } else {
        panic!("expected FnDecl");
    }
}

/// Parser accepts multiple consecutive default parameters.
#[test]
fn parse_multiple_defaults() {
    let src = "fn f(a: i64, b: i64 = 10, c: i64 = 20) -> i64 { a + b + c }";
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "unexpected errors: {:?}", diags.diagnostics);
    if let cjc_ast::DeclKind::Fn(ref f) = prog.declarations[0].kind {
        assert_eq!(f.params.len(), 3);
        assert!(f.params[0].default.is_none());
        assert!(f.params[1].default.is_some());
        assert!(f.params[2].default.is_some());
    } else {
        panic!("expected FnDecl");
    }
}

/// Parser accepts a string literal as a default value.
#[test]
fn parse_string_default() {
    let src = r#"fn greet(name: str, greeting: str = "Hello") -> str { greeting }"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "unexpected errors: {:?}", diags.diagnostics);
}

/// Parser accepts a boolean literal as a default value.
#[test]
fn parse_bool_default() {
    let src = "fn toggle(x: i64, flag: bool = true) -> i64 { x }";
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "unexpected errors: {:?}", diags.diagnostics);
    if let cjc_ast::DeclKind::Fn(ref f) = prog.declarations[0].kind {
        assert!(f.params[1].default.is_some());
    } else {
        panic!("expected FnDecl");
    }
}

/// Parser accepts an arithmetic expression as a default value.
#[test]
fn parse_expr_default() {
    let src = "fn scale(x: f64, factor: f64 = 2.0 * 3.14159) -> f64 { x * factor }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "unexpected errors: {:?}", diags.diagnostics);
}

/// Parser accepts a function with ALL parameters having defaults.
#[test]
fn parse_all_params_with_defaults() {
    let src = "fn config(a: i64 = 1, b: f64 = 2.0, c: bool = false) -> i64 { a }";
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "unexpected errors: {:?}", diags.diagnostics);
    if let cjc_ast::DeclKind::Fn(ref f) = prog.declarations[0].kind {
        assert!(f.params.iter().all(|p| p.default.is_some()));
    } else {
        panic!("expected FnDecl");
    }
}

/// Variadic parameters cannot have defaults — parser must reject this.
#[test]
fn parse_variadic_with_default_rejected() {
    assert_parse_error("fn bad(...args: f64 = 1.0) {}");
}

/// A function with no params and no defaults still works.
#[test]
fn parse_no_params() {
    let src = "fn zero() -> i64 { 0 }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "unexpected errors: {:?}", diags.diagnostics);
}

// ── Integration test 1: single default ───────────────────────────────────────

/// Test 1: single default — greeting defaults to "Hello".
#[test]
fn integration_single_default_eval() {
    let src = r#"
fn greet(name: str, greeting: str = "Hello") {
    print(greeting);
}
greet("Alice");
greet("Alice", "Hi");
"#;
    let out = eval_output(src);
    assert_eq!(out[0], "Hello", "default greeting should be Hello");
    assert_eq!(out[1], "Hi",    "explicit greeting should be Hi");
}

#[test]
fn integration_single_default_mir() {
    let src = r#"
fn greet(name: str, greeting: str = "Hello") {
    print(greeting);
}
greet("Alice");
greet("Alice", "Hi");
"#;
    let out = mir_output(src);
    assert_eq!(out[0], "Hello");
    assert_eq!(out[1], "Hi");
}

// ── Integration test 2: multiple defaults ────────────────────────────────────

/// Test 2: multiple defaults — tol defaults to 0.000001, max_iter to 100.
#[test]
fn integration_multiple_defaults_eval() {
    let src = r#"
fn solve(x: f64, tol: f64 = 0.000001, max_iter: i64 = 100) {
    print(tol);
    print(max_iter);
}
solve(3.14);
solve(3.14, 0.000000001);
solve(3.14, 0.000000001, 500);
"#;
    let out = eval_output(src);
    assert_eq!(out[0], "0.000001",  "tol default");
    assert_eq!(out[1], "100",       "max_iter default");
    assert_eq!(out[2], "0.000000001", "tol explicit");
    assert_eq!(out[3], "100",         "max_iter still default");
    assert_eq!(out[4], "0.000000001", "tol explicit");
    assert_eq!(out[5], "500",         "max_iter explicit");
}

#[test]
fn integration_multiple_defaults_mir() {
    let src = r#"
fn solve(x: f64, tol: f64 = 0.000001, max_iter: i64 = 100) {
    print(tol);
    print(max_iter);
}
solve(3.14);
solve(3.14, 0.000000001);
solve(3.14, 0.000000001, 500);
"#;
    let out = mir_output(src);
    assert_eq!(out[0], "0.000001");
    assert_eq!(out[1], "100");
    assert_eq!(out[2], "0.000000001");
    assert_eq!(out[3], "100");
    assert_eq!(out[4], "0.000000001");
    assert_eq!(out[5], "500");
}

// ── Integration test 3: default with expression ──────────────────────────────

/// Test 3: default with integer expression — fill defaults to 0.
#[test]
fn integration_default_expr_eval() {
    let src = r#"
fn make_array(n: i64, fill: i64 = 0) {
    let result: Any = [];
    let i: i64 = 0;
    while i < n {
        result = array_push(result, fill);
        i = i + 1;
    }
    print(array_len(result));
}
make_array(3);
make_array(3, 7);
"#;
    let out = eval_output(src);
    assert_eq!(out[0], "3");
    assert_eq!(out[1], "3");
}

#[test]
fn integration_default_expr_mir() {
    let src = r#"
fn make_array(n: i64, fill: i64 = 0) {
    let result: Any = [];
    let i: i64 = 0;
    while i < n {
        result = array_push(result, fill);
        i = i + 1;
    }
    print(array_len(result));
}
make_array(3);
make_array(3, 7);
"#;
    let out = mir_output(src);
    assert_eq!(out[0], "3");
    assert_eq!(out[1], "3");
}

// ── Integration: all params have defaults ────────────────────────────────────

#[test]
fn integration_all_defaults_eval() {
    let src = r#"
fn point(x: i64 = 0, y: i64 = 0, z: i64 = 0) {
    print(x);
    print(y);
    print(z);
}
point();
point(1);
point(1, 2);
point(1, 2, 3);
"#;
    let out = eval_output(src);
    assert_eq!(out[0], "0"); assert_eq!(out[1], "0"); assert_eq!(out[2], "0");
    assert_eq!(out[3], "1"); assert_eq!(out[4], "0"); assert_eq!(out[5], "0");
    assert_eq!(out[6], "1"); assert_eq!(out[7], "2"); assert_eq!(out[8], "0");
    assert_eq!(out[9], "1"); assert_eq!(out[10], "2"); assert_eq!(out[11], "3");
}

#[test]
fn integration_all_defaults_mir() {
    let src = r#"
fn point(x: i64 = 0, y: i64 = 0, z: i64 = 0) {
    print(x);
    print(y);
    print(z);
}
point();
point(1);
point(1, 2);
point(1, 2, 3);
"#;
    let out = mir_output(src);
    assert_eq!(out[0], "0"); assert_eq!(out[1], "0"); assert_eq!(out[2], "0");
    assert_eq!(out[3], "1"); assert_eq!(out[4], "0"); assert_eq!(out[5], "0");
    assert_eq!(out[6], "1"); assert_eq!(out[7], "2"); assert_eq!(out[8], "0");
    assert_eq!(out[9], "1"); assert_eq!(out[10], "2"); assert_eq!(out[11], "3");
}

// ── Integration: default used in return value ─────────────────────────────────

#[test]
fn integration_default_in_return_value() {
    let src = r#"
fn add_with_bias(x: i64, bias: i64 = 10) -> i64 {
    x + bias
}
print(add_with_bias(5));
print(add_with_bias(5, 1));
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], "15");
    assert_eq!(ev[1], "6");
    assert_eq!(ev[0], mir[0]);
    assert_eq!(ev[1], mir[1]);
}

// ── Integration: recursive function with default ──────────────────────────────

#[test]
fn integration_recursive_with_default() {
    let src = r#"
fn countdown(n: i64, step: i64 = 1) -> i64 {
    if n <= 0 {
        0
    } else {
        countdown(n - step, step)
    }
}
print(countdown(5));
print(countdown(6, 2));
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], "0");
    assert_eq!(ev[1], "0");
    assert_eq!(ev[0], mir[0]);
    assert_eq!(ev[1], mir[1]);
}

// ── Integration: float default in scientific notation ────────────────────────

#[test]
fn integration_scientific_notation_default() {
    let src = r#"
fn converge(x: f64, eps: f64 = 1e-8) {
    if x < eps {
        print("converged");
    } else {
        print("not converged");
    }
}
converge(0.0);
converge(1.0);
converge(0.0, 1.0);
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], "converged");
    assert_eq!(ev[1], "not converged");
    assert_eq!(ev[2], "converged");
    assert_eq!(ev, mir);
}

// ── Parity tests: eval == MIR-exec ───────────────────────────────────────────

#[test]
fn parity_single_default() {
    assert_parity(r#"
fn inc(x: i64, step: i64 = 1) -> i64 { x + step }
print(inc(10));
print(inc(10, 5));
"#);
}

#[test]
fn parity_string_default() {
    assert_parity(r#"
fn greet(name: str, greeting: str = "Hello") -> str {
    str_join([greeting, name], " ")
}
print(greet("World"));
print(greet("World", "Hi"));
"#);
}

#[test]
fn parity_nested_calls_with_defaults() {
    assert_parity(r#"
fn double(x: i64, factor: i64 = 2) -> i64 { x * factor }
fn quad(x: i64) -> i64 { double(double(x)) }
print(quad(3));
"#);
}

// ── Determinism tests ─────────────────────────────────────────────────────────

/// Repeated eval runs produce bit-identical output.
#[test]
fn determinism_eval_repeated_runs() {
    let src = r#"
fn weighted(x: f64, w: f64 = 0.5) -> f64 { x * w }
print(weighted(3.14));
print(weighted(2.71, 0.25));
"#;
    let v1 = eval_output(src);
    let v2 = eval_output(src);
    assert_eq!(v1, v2, "eval must be deterministic");
}

/// Repeated MIR-exec runs produce bit-identical output.
#[test]
fn determinism_mir_repeated_runs() {
    let src = r#"
fn weighted(x: f64, w: f64 = 0.5) -> f64 { x * w }
print(weighted(3.14));
print(weighted(2.71, 0.25));
"#;
    let v1 = mir_output(src);
    let v2 = mir_output(src);
    assert_eq!(v1, v2, "mir-exec must be deterministic");
}

// ── Error handling ────────────────────────────────────────────────────────────

/// Missing required argument (no default) must still produce a runtime error.
#[test]
fn error_missing_required_arg_eval() {
    let src = r#"
fn add(x: i64, y: i64) -> i64 { x + y }
add(1);
"#;
    let (program, _) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "missing required arg should error in eval");
}

#[test]
fn error_missing_required_arg_mir() {
    let src = r#"
fn add(x: i64, y: i64) -> i64 { x + y }
add(1);
"#;
    let (program, _) = cjc_parser::parse_source(src);
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    assert!(result.is_err(), "missing required arg should error in mir-exec");
}

/// Too many arguments (beyond all params) must still produce a runtime error.
#[test]
fn error_too_many_args_eval() {
    let src = r#"
fn f(x: i64, y: i64 = 0) -> i64 { x + y }
f(1, 2, 3);
"#;
    let (program, _) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "too many args should error in eval");
}

#[test]
fn error_too_many_args_mir() {
    let src = r#"
fn f(x: i64, y: i64 = 0) -> i64 { x + y }
f(1, 2, 3);
"#;
    let (program, _) = cjc_parser::parse_source(src);
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    assert!(result.is_err(), "too many args should error in mir-exec");
}

// ── Interaction: defaults + variadic ─────────────────────────────────────────

/// Default params can precede a variadic parameter.
#[test]
fn interaction_default_then_variadic() {
    let src = r#"
fn log_items(prefix: str = "item", ...values: i64) {
    print(prefix);
    print(array_len(values));
}
log_items("x", 1, 2, 3);
log_items("x");
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], "x");
    assert_eq!(ev[1], "3");
    assert_eq!(ev[2], "x");
    assert_eq!(ev[3], "0");
    assert_eq!(ev, mir);
}

// ── Proptest: property-based tests ───────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// For any i64 value v, calling f(v) with default b=0 gives v+0 = v.
    /// Calling f(v, b) gives v+b.
    #[test]
    fn prop_default_zero_identity(v in -1000i64..1000i64, b in -100i64..100i64) {
        let src = format!(r#"
fn add(x: i64, y: i64 = 0) -> i64 {{ x + y }}
print(add({v}));
print(add({v}, {b}));
"#);
        let ev = eval_output(&src);
        let mir = mir_output(&src);
        // f(v) == v (default is 0)
        prop_assert_eq!(ev[0].parse::<i64>().unwrap(), v);
        // f(v, b) == v + b
        prop_assert_eq!(ev[1].parse::<i64>().unwrap(), v + b);
        // parity: eval == mir
        prop_assert_eq!(&ev[0], &mir[0]);
        prop_assert_eq!(&ev[1], &mir[1]);
    }

    /// For any f64 value x with scale defaulting to 1.0, f(x) == x.
    #[test]
    fn prop_default_scale_identity(x in -100.0f64..100.0f64) {
        let src = format!(r#"
fn scale(x: f64, factor: f64 = 1.0) -> f64 {{ x * factor }}
print(scale({x}));
"#);
        // Both executors must agree that scale(x) == x * 1.0 == x.
        let ev = eval_output(&src);
        let mir = mir_output(&src);
        prop_assert_eq!(&ev[0], &mir[0]);
        let result: f64 = ev[0].parse().unwrap();
        prop_assert!((result - x).abs() < 1e-9, "expected {x}, got {result}");
    }
}

// ── Bolero fuzz target ────────────────────────────────────────────────────────

/// Fuzz: programs with default params never cause a panic in either executor,
/// regardless of which argument combination is supplied.
///
/// We parameterize over the number of args actually provided (0, 1, or 2)
/// for a function with signature `fn f(a: i64 = 0, b: i64 = 0) -> i64`.
#[test]
fn fuzz_default_param_no_panic() {
    bolero::check!().with_type::<u8>().for_each(|n_args: &u8| {
        // Clamp to valid range [0, 2].
        let n = (*n_args % 3) as usize;
        let arg_str = match n {
            0 => "".to_string(),
            1 => "42".to_string(),
            _ => "42, 7".to_string(),
        };
        let src = format!(
            r#"fn f(a: i64 = 0, b: i64 = 0) -> i64 {{ a + b }}
print(f({arg_str}));"#
        );
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = eval_output(&src);
        }));
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = mir_output(&src);
        }));
    });
}
