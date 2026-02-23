//! H-1: Span-aware unification — `unify_spanned()` exists and emits spanned diagnostics.

use cjc_diag::{DiagnosticBag, Span};
use cjc_types::{unify, unify_spanned, Type, TypeSubst};

/// Test 1: `unify_spanned` succeeds on identical types — no diagnostic emitted.
#[test]
fn test_unify_spanned_success_no_diagnostic() {
    let mut subst = TypeSubst::new();
    let mut diag = DiagnosticBag::new();
    let span = Span::new(0, 10);
    let result = unify_spanned(&Type::I64, &Type::I64, &mut subst, span, &mut diag);
    assert_eq!(result, Type::I64);
    assert!(!diag.has_errors(), "successful unify should emit no errors");
}

/// Test 2: `unify_spanned` on mismatched types emits a spanned error and returns Type::Error.
#[test]
fn test_unify_spanned_failure_emits_spanned_error() {
    let mut subst = TypeSubst::new();
    let mut diag = DiagnosticBag::new();
    let span = Span::new(5, 20);
    let result = unify_spanned(&Type::I64, &Type::F64, &mut subst, span, &mut diag);
    assert!(result.is_error(), "failed unify should return Type::Error");
    assert!(diag.has_errors(), "failed unify should emit an error diagnostic");
    let err = &diag.diagnostics[0];
    // The error should carry the span we passed
    assert_eq!(err.span.start, 5);
    assert_eq!(err.span.end, 20);
    assert_eq!(err.code, "E0100");
}

/// Test 3: `unify_spanned` with a type variable — should succeed and bind the variable.
#[test]
fn test_unify_spanned_binds_type_variable() {
    use cjc_types::TypeVarId;
    let mut subst = TypeSubst::new();
    let mut diag = DiagnosticBag::new();
    let span = Span::new(0, 5);
    let tv = Type::Var(TypeVarId(0));
    let result = unify_spanned(&tv, &Type::F64, &mut subst, span, &mut diag);
    assert_eq!(result, Type::F64);
    assert!(!diag.has_errors());
    // Type variable 0 should now be bound to F64
    assert_eq!(subst.get(&TypeVarId(0)), Some(&Type::F64));
}

/// Test 4: Original `unify()` still works (backward compatibility).
#[test]
fn test_original_unify_still_works() {
    let mut subst = TypeSubst::new();
    assert!(unify(&Type::I64, &Type::I64, &mut subst).is_ok());
    assert!(unify(&Type::I64, &Type::F64, &mut subst).is_err());
}

/// Test 5: Multiple span-aware unification errors accumulate in DiagnosticBag.
#[test]
fn test_unify_spanned_multiple_errors_accumulate() {
    let mut subst = TypeSubst::new();
    let mut diag = DiagnosticBag::new();

    unify_spanned(&Type::I64, &Type::F64, &mut subst, Span::new(0, 5), &mut diag);
    unify_spanned(&Type::Bool, &Type::Str, &mut subst, Span::new(10, 15), &mut diag);

    assert_eq!(diag.error_count(), 2, "both errors should accumulate");
    assert_eq!(diag.diagnostics[0].span.start, 0);
    assert_eq!(diag.diagnostics[1].span.start, 10);
}

/// Test 6: Non-generic function call with wrong arg type emits spanned error.
#[test]
fn test_type_checker_non_generic_call_type_mismatch_emits_error() {
    use cjc_parser::parse_source;
    use cjc_types::TypeChecker;
    // sum_ints expects (i64, i64) but we pass (f64, i64)
    let src = r#"
fn sum_ints(a: i64, b: i64) -> i64 {
    a + b
}
fn main() -> i64 {
    sum_ints(1.5, 2)
}
"#;
    let (prog, parse_diags) = parse_source(src);
    assert!(!parse_diags.has_errors(), "should parse clean");
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    assert!(
        checker.diagnostics.has_errors(),
        "passing f64 to i64 param should be a type error"
    );
}

/// Test 7: TypeChecker diagnostics now include span info (not zero spans) on binary op errors.
#[test]
fn test_type_checker_binary_op_error_has_nonzero_span() {
    use cjc_parser::parse_source;
    use cjc_types::TypeChecker;
    let src = r#"
fn main() -> bool {
    1 + true
}
"#;
    let (prog, _) = parse_source(src);
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    assert!(checker.diagnostics.has_errors());
    // The binary op error should have a non-trivial span
    let err = checker.diagnostics.diagnostics.iter().find(|d| d.code == "E0101");
    assert!(err.is_some(), "E0101 binary op error should be emitted");
    let err = err.unwrap();
    // Parser assigns spans; the span should have start < end for real source
    assert!(
        err.span.end > err.span.start || err.span.start > 0,
        "error span should be non-trivial, got start={} end={}",
        err.span.start, err.span.end
    );
}
