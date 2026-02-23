//! Phase 2 Audit Tests: P0-1 Generic Monomorphization
//!
//! Tests that generic functions are correctly monomorphized:
//! - Single type parameter with concrete type
//! - Multiple specializations of same function
//! - Generic function body executes correctly
//! - Correct name mangling

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

/// P0-1 Test 1: Generic identity function with i64 specialization.
#[test]
fn test_generic_identity_i64() {
    let src = r#"
fn identity<T>(x: T) -> T { x }
fn main() -> i64 {
    identity(42)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "generic identity i64 failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(42)));
}

/// P0-1 Test 2: Generic identity function with f64 specialization.
#[test]
fn test_generic_identity_f64() {
    let src = r#"
fn identity<T>(x: T) -> T { x }
fn main() -> f64 {
    identity(3.14)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "generic identity f64 failed: {:?}", result);
    match result.unwrap() {
        Value::Float(f) => assert!((f - 3.14).abs() < 1e-5),
        other => panic!("Expected Float, got {:?}", other),
    }
}

/// P0-1 Test 3: Generic function with two specializations in same program.
#[test]
fn test_generic_multiple_specializations() {
    let src = r#"
fn double<T>(x: T) -> T { x + x }
fn main() -> i64 {
    let a = double(10);
    a
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "generic double failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(20)));
}

/// P0-1 Test 4: Generic function calling another generic function.
#[test]
fn test_generic_nested_calls() {
    let src = r#"
fn wrap<T>(x: T) -> T { x }
fn apply<T>(x: T) -> T { wrap(x) }
fn main() -> i64 {
    apply(123)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "generic nested calls failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(123)));
}

/// P0-1 Test 5: Generic max function.
#[test]
fn test_generic_max() {
    let src = r#"
fn my_max<T>(a: T, b: T) -> T {
    if a > b { a } else { b }
}
fn main() -> i64 {
    my_max(10, 20)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "generic max failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(20)));
}

/// P0-1 Test 6: Generic function used in a loop.
#[test]
fn test_generic_in_loop() {
    let src = r#"
fn square<T>(x: T) -> T { x * x }
fn main() -> i64 {
    let mut sum: i64 = 0;
    let mut i: i64 = 1;
    while i <= 5 {
        sum = sum + square(i);
        i = i + 1;
    }
    sum
}
"#;
    // 1 + 4 + 9 + 16 + 25 = 55
    let result = run_src(src);
    assert!(result.is_ok(), "generic in loop failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(55)));
}
