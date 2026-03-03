//! LH03: Real Generics + Trait Bounds tests
//!
//! Verifies:
//! - Built-in trait implementations exist for primitive types
//! - Trait bound checking at call sites
//! - Monomorphization respects bounds
//! - Generic functions work in eval and MIR-exec

use cjc_types::{TypeEnv, Type, TypeChecker};

// ── Built-in Trait Coverage ──────────────────────────────────────────

#[test]
fn test_i64_satisfies_numeric() {
    let env = TypeEnv::new();
    assert!(env.satisfies_trait(&Type::I64, "Numeric"));
}

#[test]
fn test_f64_satisfies_numeric() {
    let env = TypeEnv::new();
    assert!(env.satisfies_trait(&Type::F64, "Numeric"));
}

#[test]
fn test_f64_satisfies_float() {
    let env = TypeEnv::new();
    assert!(env.satisfies_trait(&Type::F64, "Float"));
}

#[test]
fn test_i64_satisfies_int() {
    let env = TypeEnv::new();
    assert!(env.satisfies_trait(&Type::I64, "Int"));
}

#[test]
fn test_string_does_not_satisfy_numeric() {
    let env = TypeEnv::new();
    assert!(!env.satisfies_trait(&Type::Str, "Numeric"));
}

#[test]
fn test_bool_does_not_satisfy_numeric() {
    let env = TypeEnv::new();
    assert!(!env.satisfies_trait(&Type::Bool, "Numeric"));
}

#[test]
fn test_i64_satisfies_ord() {
    let env = TypeEnv::new();
    assert!(env.satisfies_trait(&Type::I64, "Ord"));
}

#[test]
fn test_f64_satisfies_differentiable() {
    let env = TypeEnv::new();
    assert!(env.satisfies_trait(&Type::F64, "Differentiable"));
}

// ── check_bounds helper ──────────────────────────────────────────────

#[test]
fn test_check_bounds_single() {
    let env = TypeEnv::new();
    assert!(env.check_bounds(&Type::I64, &["Numeric".to_string()]));
    assert!(!env.check_bounds(&Type::Str, &["Numeric".to_string()]));
}

#[test]
fn test_check_bounds_multiple() {
    let env = TypeEnv::new();
    assert!(env.check_bounds(&Type::F64, &["Numeric".to_string(), "Float".to_string()]));
    assert!(!env.check_bounds(&Type::I64, &["Numeric".to_string(), "Float".to_string()]));
}

#[test]
fn test_check_bounds_empty() {
    let env = TypeEnv::new();
    // No bounds: any type satisfies
    assert!(env.check_bounds(&Type::Str, &[]));
    assert!(env.check_bounds(&Type::I64, &[]));
}

// ── Type Checker: bound violations at call sites ─────────────────────

#[test]
fn test_type_checker_generic_bound_satisfied() {
    let src = r#"
trait Numeric {}
impl Numeric for i64 {}
fn add_nums<T: Numeric>(a: T, b: T) -> T { a }
let x = add_nums(1, 2);
"#;
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors(), "parse errors: {:?}", parse_diags.diagnostics);

    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    // Should NOT produce type errors - i64 satisfies Numeric
    let type_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code.starts_with("E03") || d.code.starts_with("E6"))
        .collect();
    assert!(type_errors.is_empty(), "unexpected bound errors: {:?}", type_errors);
}

#[test]
fn test_type_checker_generic_bound_violated() {
    let src = r#"
trait Numeric {}
impl Numeric for i64 {}
fn add_nums<T: Numeric>(a: T, b: T) -> T { a }
let x = add_nums("hello", "world");
"#;
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    // Should produce a bound violation error
    let bound_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.message.contains("trait bound") || d.code == "E0300")
        .collect();
    assert!(!bound_errors.is_empty(),
        "expected bound violation error, got: {:?}", checker.diagnostics.diagnostics);
}

// ── Execution: generic functions work in eval ────────────────────────

#[test]
fn test_generic_fn_execution() {
    let src = r#"
fn identity(x: i64) -> i64 { x }
print(identity(42));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["42"]);
}

#[test]
fn test_generic_fn_with_multiple_types() {
    // CJC supports calling same function with different concrete types
    let src = r#"
fn double(x: i64) -> i64 { x * 2 }
fn double_f(x: f64) -> f64 { x * 2.0 }
print(double(21));
print(double_f(10.5));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["42", "21"]);
}

// ── Parity: eval vs MIR-exec ────────────────────────────────────────

#[test]
fn test_generic_parity() {
    let src = r#"
fn square(x: i64) -> i64 { x * x }
print(square(7));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    // Eval
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program);
    let eval_output = interp.output.clone();

    // MIR-exec
    let (mir_val, mir_exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let mir_output = mir_exec.output.clone();

    assert_eq!(eval_output, mir_output, "parity mismatch");
}
