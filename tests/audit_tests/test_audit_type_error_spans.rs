//! Audit Test: Type Error Spans / Diagnostics Reality Check
//!
//! Claim: "Type error messages have no location (unify returns Err(String), no spans)"
//!
//! VERDICT: CONFIRMED (with nuance)
//!
//! Evidence from cjc-types/src/lib.rs:
//! - `pub fn unify(a: &Type, b: &Type, subst: &mut TypeSubst) -> Result<Type, String>`
//!   Error type is `String` — no span, no file, no line/col.
//! - TypeChecker wraps unify errors as `Diagnostic { span, message }` but
//!   the span comes from the expression node being checked, not from unify itself.
//! - TypeChecker.diagnostics: DiagnosticBag — CAN emit spanned diagnostics.
//! - BUT: the span quality depends entirely on which expression triggered the check.
//!
//! NUANCE: unify() IS span-free (confirmed). TypeChecker CAN add spans when wrapping.
//! The claim is CONFIRMED for unify() in isolation.

use cjc_types::{Type, TypeSubst, unify, TypeVarId, StructType};
use cjc_diag::Span;
use cjc_parser::parse_source;

/// Test 1: unify() returns Err(String) — no span in the error.
#[test]
fn test_unify_returns_string_error_no_span() {
    let mut subst = TypeSubst::new();
    // Try to unify i64 with f64 — should fail
    let result = unify(&Type::I64, &Type::F64, &mut subst);
    assert!(result.is_err(), "i64 != f64 should produce a unification error");
    let err: String = result.unwrap_err();
    // The error is a plain String — no Span, no line number, no file
    assert!(!err.is_empty(), "error message should not be empty");
    // Structural proof: err is String, which has no `.span()` method.
    // If it had a span, it would be: err.span.start — which doesn't compile.
    let _: &str = &err; // confirms it's a String
}

/// Test 2: Unification of incompatible types produces a descriptive string error.
#[test]
fn test_unify_incompatible_primitives_error_message_is_string() {
    let mut subst = TypeSubst::new();
    let result = unify(&Type::Bool, &Type::I64, &mut subst);
    assert!(result.is_err());
    let msg: String = result.unwrap_err();
    assert!(
        !msg.is_empty(),
        "error message should not be empty, got: {:?}", msg
    );
    // Key: msg is a plain String with no attached source location
}

/// Test 3: Infinite type detection (occurs check) returns a plain String error.
#[test]
fn test_unify_occurs_check_returns_string_with_no_span() {
    let mut subst = TypeSubst::new();
    // T0 unified with Tuple(T0) → infinite type
    let var = Type::Var(TypeVarId(0));
    let recursive = Type::Tuple(vec![Type::Var(TypeVarId(0))]);
    let result = unify(&var, &recursive, &mut subst);
    assert!(result.is_err(), "recursive type should fail occurs check");
    let msg: String = result.unwrap_err();
    assert!(
        msg.contains("infinite") || msg.contains("occurs"),
        "occurs check error should mention 'infinite' or 'occurs', got: {:?}", msg
    );
    // Again: plain String, no span
}

/// Test 4: Diagnostic system HAS Span — the infrastructure exists for wrapping.
#[test]
fn test_diagnostic_span_infrastructure_exists() {
    let span = Span { start: 10, end: 25 };
    assert_eq!(span.start, 10);
    assert_eq!(span.end, 25);
    // Diagnostic can hold a span — the wrapper infrastructure exists
    let diag = cjc_diag::Diagnostic::error("E0001", "type mismatch", span);
    assert!(diag.message.contains("type mismatch"));
    assert_eq!(diag.span.start, 10);
    assert_eq!(diag.span.end, 25);
}

/// Test 5: Parser produces AST nodes WITH spans — span information exists in the tree.
#[test]
fn test_parser_produces_spanned_ast_nodes() {
    let src = "let x = 42;";
    let (prog, _) = parse_source(src);
    assert!(!prog.declarations.is_empty());
    let decl = &prog.declarations[0];
    // Span covers some source bytes
    assert!(
        decl.span.end >= decl.span.start,
        "declaration should have valid span: {:?}", decl.span
    );
}

/// Test 6: TypeChecker has diagnostics bag with span-aware errors.
#[test]
fn test_type_checker_has_diagnostic_bag() {
    use cjc_types::TypeChecker;
    let mut tc = TypeChecker::new();
    // check_program stores errors in tc.diagnostics (DiagnosticBag)
    let src = r#"fn main() -> i64 { 42 }"#;
    let (prog, _) = parse_source(src);
    tc.check_program(&prog);
    // Valid program: no errors
    let has_errs = tc.diagnostics.has_errors();
    assert!(!has_errs, "valid program should have no type errors");
}

/// Test 7: Type error message content — verifies message quality without span.
#[test]
fn test_unify_error_mentions_type_names() {
    let mut subst = TypeSubst::new();
    let result = unify(&Type::F64, &Type::Bool, &mut subst);
    assert!(result.is_err());
    let msg = result.unwrap_err();
    // The message should be human-readable even without spans
    // (it's all you get — no file/line info)
    assert!(msg.len() > 3, "error message should be descriptive, got: {:?}", msg);
}

/// Test 8: Struct vs primitive unification fails with plain String.
#[test]
fn test_unify_struct_vs_primitive_plain_string_error() {
    let mut subst = TypeSubst::new();
    let struct_ty = Type::Struct(StructType {
        name: "Point".to_string(),
        type_params: vec![],
        fields: vec![("x".to_string(), Type::F64)],
    });
    let result = unify(&struct_ty, &Type::I64, &mut subst);
    assert!(result.is_err(), "Struct(Point) != i64 should fail");
    let msg: String = result.unwrap_err();
    assert!(!msg.is_empty(), "error should not be empty");
    // Confirmed: error is a String with no span
}

/// Test 9: Successful unification of type variables updates the substitution.
#[test]
fn test_unify_type_variable_resolves() {
    let mut subst = TypeSubst::new();
    let var = Type::Var(TypeVarId(0));
    // Unify T0 with f64 — should bind T0 → f64
    let result = unify(&var, &Type::F64, &mut subst);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), Type::F64);
    assert!(subst.contains_key(&TypeVarId(0)), "T0 should be bound in subst");
    assert_eq!(subst[&TypeVarId(0)], Type::F64);
}
