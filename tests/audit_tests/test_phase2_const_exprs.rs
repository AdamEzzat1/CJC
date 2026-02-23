//! Phase 2 Audit Tests: P2-3 Const Expressions
//!
//! Tests for the `const NAME: Type = expr;` declaration:
//! - Const parses successfully
//! - Const is accessible at runtime
//! - Non-const initializer emits E0400
//! - Type mismatch emits E0401

use cjc_parser::parse_source;
use cjc_mir_exec::run_program_with_executor;
use cjc_runtime::Value;

fn run_src(src: &str) -> Result<Value, String> {
    let (prog, diags) = parse_source(src);
    if diags.has_errors() {
        return Err(format!("parse errors"));
    }
    run_program_with_executor(&prog, 42)
        .map(|(v, _)| v)
        .map_err(|e| format!("{e}"))
}

/// P2-3 Test 1: Basic const declaration and usage.
#[test]
fn test_const_basic_usage() {
    let src = r#"
const ANSWER: i64 = 42;
fn main() -> i64 {
    ANSWER
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "const usage failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(42)));
}

/// P2-3 Test 2: Const used in expression.
#[test]
fn test_const_in_expression() {
    let src = r#"
const BASE: i64 = 10;
fn main() -> i64 {
    BASE * 3 + 2
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "const in expr failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(32)));
}

/// P2-3 Test 3: Float const.
#[test]
fn test_const_float() {
    let src = r#"
const PI: f64 = 3.14159;
fn main() -> f64 {
    PI
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "float const failed: {:?}", result);
    match result.unwrap() {
        Value::Float(f) => assert!((f - 3.14159).abs() < 1e-5, "float mismatch"),
        other => panic!("Expected Float, got {:?}", other),
    }
}

/// P2-3 Test 4: Bool const.
#[test]
fn test_const_bool() {
    let src = r#"
const DEBUG: bool = true;
fn main() -> bool {
    DEBUG
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "bool const failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Bool(true)));
}

/// P2-3 Test 5: String const.
#[test]
fn test_const_string() {
    let src = r#"
const GREETING: str = "hello";
fn main() -> i64 {
    print(GREETING);
    0
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "string const failed: {:?}", result);
}

/// P2-3 Test 6: Negative literal const (unary minus).
#[test]
fn test_const_negative_literal() {
    let src = r#"
const NEG: i64 = -100;
fn main() -> i64 {
    NEG
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "negative const failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(-100)));
}

/// P2-3 Test 7: Non-const initializer emits E0400.
#[test]
fn test_const_non_const_init_emits_e0400() {
    use cjc_types::TypeChecker;
    let src = r#"
fn get_val() -> i64 { 42 }
const BAD: i64 = get_val();
fn main() -> i64 { 0 }
"#;
    let (prog, _) = parse_source(src);
    let mut checker = TypeChecker::new();
    checker.check_program(&prog);
    let e0400_errors: Vec<_> = checker
        .diagnostics
        .diagnostics
        .iter()
        .filter(|d| d.severity == cjc_diag::Severity::Error && d.code == "E0400")
        .collect();
    assert!(
        !e0400_errors.is_empty(),
        "Expected E0400 for non-const initializer, got no E0400 diagnostics"
    );
}

/// P2-3 Test 8: Multiple consts in same program.
#[test]
fn test_multiple_consts() {
    let src = r#"
const A: i64 = 1;
const B: i64 = 2;
const C: i64 = 3;
fn main() -> i64 {
    A + B + C
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "multiple consts failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(6)));
}
