//! LH02: Type Inference tests
//!
//! Verifies:
//! - InferCtx constraint solving
//! - Local type inference for let bindings (already working)
//! - Return type inference from function body
//! - Bidirectional inference through unification
//! - Backward compatibility: explicitly annotated code unchanged

use cjc_types::inference::InferCtx;
use cjc_types::{Type, TypeVarId};
use cjc_diag::Span;

// ── InferCtx Unit Tests ──────────────────────────────────────────────

#[test]
fn test_infer_ctx_basic() {
    let mut ctx = InferCtx::new(100);
    let v = ctx.fresh_var();
    assert_eq!(v, Type::Var(TypeVarId(100)));
    assert_eq!(ctx.next_var_counter(), 101);
}

#[test]
fn test_infer_ctx_solve_simple() {
    let mut ctx = InferCtx::new(0);
    let tv = ctx.fresh_var();
    ctx.constrain(tv.clone(), Type::F64, Span::new(0, 1));
    let errors = ctx.solve();
    assert!(errors.is_empty());
    assert_eq!(ctx.apply(&tv), Type::F64);
}

#[test]
fn test_infer_ctx_solve_transitive() {
    let mut ctx = InferCtx::new(0);
    let a = ctx.fresh_var();
    let b = ctx.fresh_var();
    ctx.constrain(a.clone(), b.clone(), Span::new(0, 1));
    ctx.constrain(b.clone(), Type::Str, Span::new(2, 3));
    let errors = ctx.solve();
    assert!(errors.is_empty());
    assert_eq!(ctx.apply(&a), Type::Str);
}

#[test]
fn test_infer_ctx_solve_conflict() {
    let mut ctx = InferCtx::new(0);
    let tv = ctx.fresh_var();
    ctx.constrain(tv.clone(), Type::I64, Span::new(0, 1));
    ctx.constrain(tv.clone(), Type::Str, Span::new(2, 3));
    let errors = ctx.solve();
    assert!(!errors.is_empty());
    assert!(errors[0].code == "E2004");
}

#[test]
fn test_infer_ctx_tuple_unification() {
    let mut ctx = InferCtx::new(0);
    let tv = ctx.fresh_var();
    ctx.constrain(
        Type::Tuple(vec![tv.clone(), Type::Bool]),
        Type::Tuple(vec![Type::I64, Type::Bool]),
        Span::new(0, 5),
    );
    let errors = ctx.solve();
    assert!(errors.is_empty());
    assert_eq!(ctx.apply(&tv), Type::I64);
}

#[test]
fn test_infer_ctx_function_type() {
    let mut ctx = InferCtx::new(0);
    let ret_var = ctx.fresh_var();
    ctx.constrain(
        Type::Fn { params: vec![Type::I64], ret: Box::new(ret_var.clone()) },
        Type::Fn { params: vec![Type::I64], ret: Box::new(Type::Bool) },
        Span::new(0, 1),
    );
    let errors = ctx.solve();
    assert!(errors.is_empty());
    assert_eq!(ctx.apply(&ret_var), Type::Bool);
}

#[test]
fn test_infer_ctx_resolve_final_defaults() {
    let ctx = InferCtx::new(0);
    // Unresolved type variable defaults to i64
    assert_eq!(ctx.resolve_final(&Type::Var(TypeVarId(0))), Type::I64);
    // Known types unchanged
    assert_eq!(ctx.resolve_final(&Type::F64), Type::F64);
    assert_eq!(ctx.resolve_final(&Type::Bool), Type::Bool);
}

// ── Integration: let binding inference (already works) ───────────────

#[test]
fn test_let_infer_int() {
    let src = "let x = 42; print(x);";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    // Should execute without type errors
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok(), "exec error: {:?}", result);
    assert_eq!(interp.output, vec!["42"]);
}

#[test]
fn test_let_infer_float() {
    let src = "let x = 3.14; print(x);";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["3.14"]);
}

#[test]
fn test_let_infer_string() {
    let src = "let s = \"hello\"; print(s);";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["hello"]);
}

#[test]
fn test_let_infer_bool() {
    let src = "let b = true; print(b);";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["true"]);
}

#[test]
fn test_let_infer_from_expr() {
    let src = "let a = 10; let b = a + 20; print(b);";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["30"]);
}

// ── Integration: function return type inference ──────────────────────

#[test]
fn test_fn_infer_return_type() {
    // Function without explicit return type
    let src = r#"
fn double(x: i64) {
    x * 2
}
print(double(21));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok(), "exec error: {:?}", result);
    assert_eq!(interp.output, vec!["42"]);
}

#[test]
fn test_fn_explicit_return_type_still_works() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 {
    a + b
}
print(add(10, 20));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["30"]);
}

// ── Type checker: inference produces correct types ───────────────────

#[test]
fn test_type_checker_let_inference() {
    let src = "let x = 42;";
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors());

    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);
    // Should not produce type errors
    assert!(!checker.diagnostics.has_errors(),
        "type errors: {:?}", checker.diagnostics.diagnostics);
}

#[test]
fn test_type_checker_annotated_let() {
    let src = "let x: i64 = 42;";
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors());

    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);
    assert!(!checker.diagnostics.has_errors());
}

#[test]
fn test_type_checker_mismatch_still_caught() {
    let src = "let x: i64 = \"hello\";";
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors());

    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);
    // This should produce a type error
    assert!(checker.diagnostics.has_errors());
}
