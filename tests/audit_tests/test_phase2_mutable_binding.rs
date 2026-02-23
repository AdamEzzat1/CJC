//! Phase 2 Audit Tests: P0-3 Mutable Binding Enforcement (E0150)
//!
//! These tests verify that:
//! - Assigning to an immutable binding emits E0150
//! - `let mut` bindings allow reassignment
//! - Shadowed bindings behave correctly
//! - Nested scope bindings enforce mutability independently

use cjc_parser::parse_source;
use cjc_types::TypeChecker;

fn type_check_source(src: &str) -> Vec<String> {
    let (prog, _) = parse_source(src);
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    checker
        .diagnostics
        .diagnostics
        .iter()
        .filter(|d| d.severity == cjc_diag::Severity::Error)
        .map(|d| format!("{}: {}", d.code, d.message))
        .collect()
}

fn has_error(errors: &[String], code: &str) -> bool {
    errors.iter().any(|e| e.starts_with(code))
}

/// P0-3 Test 1: Assigning to an immutable let binding emits E0150.
#[test]
fn test_immutable_assign_emits_e0150() {
    let src = r#"
fn main() -> i64 {
    let x: i64 = 10;
    x = 20;
    0
}
"#;
    let errors = type_check_source(src);
    assert!(
        has_error(&errors, "E0150"),
        "Expected E0150 for immutable assignment, got: {:?}",
        errors
    );
}

/// P0-3 Test 2: Assigning to a `let mut` binding should succeed (no E0150).
#[test]
fn test_mutable_assign_no_error() {
    let src = r#"
fn main() -> i64 {
    let mut x: i64 = 10;
    x = 20;
    x
}
"#;
    let errors = type_check_source(src);
    let e0150_errors: Vec<_> = errors.iter().filter(|e| e.starts_with("E0150")).collect();
    assert!(
        e0150_errors.is_empty(),
        "Should NOT emit E0150 for mutable binding, got: {:?}",
        e0150_errors
    );
}

/// P0-3 Test 3: Function parameter assignment — must not panic.
#[test]
fn test_function_param_no_crash() {
    // Parameters are not registered through let, so assigning to them
    // may or may not error — but must not panic.
    let src = r#"
fn foo(x: i64) -> i64 {
    x = 99;
    x
}
fn main() -> i64 { foo(1) }
"#;
    let (prog, _) = parse_source(src);
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    // Must not panic — any diagnostic is acceptable.
}

/// P0-3 Test 4: Shadowed immutable binding still cannot be assigned.
#[test]
fn test_shadowed_let_is_immutable() {
    let src = r#"
fn main() -> i64 {
    let x: i64 = 1;
    let x: i64 = 2;
    x = 3;
    0
}
"#;
    // The second `let x` is also immutable, so assigning should emit E0150.
    let errors = type_check_source(src);
    assert!(
        has_error(&errors, "E0150"),
        "Expected E0150 for shadowed immutable binding, got: {:?}",
        errors
    );
}

/// P0-3 Test 5: Mutable binding accessible inside nested scope.
#[test]
fn test_mutable_binding_nested_scope_ok() {
    let src = r#"
fn main() -> i64 {
    let mut x: i64 = 10;
    if x > 5 {
        x = 99;
    }
    x
}
"#;
    let errors = type_check_source(src);
    let e0150_errors: Vec<_> = errors.iter().filter(|e| e.starts_with("E0150")).collect();
    assert!(
        e0150_errors.is_empty(),
        "Mutable binding in nested scope should not emit E0150, got: {:?}",
        e0150_errors
    );
}

/// P0-3 Test 6: Only the assigned immutable binding produces an error.
#[test]
fn test_only_assigned_binding_errors() {
    let src = r#"
fn main() -> i64 {
    let a: i64 = 1;
    let b: i64 = 2;
    b = 3;
    a
}
"#;
    let errors = type_check_source(src);
    // Should have exactly one E0150 (for `b`, not `a`)
    let e0150_count = errors.iter().filter(|e| e.starts_with("E0150")).count();
    assert_eq!(
        e0150_count, 1,
        "Expected exactly 1 E0150, got {:?}",
        errors
    );
}
