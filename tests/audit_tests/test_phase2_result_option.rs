//! Phase 2 Audit Tests: P2-6 Result/Option Builtin Methods
//!
//! Tests for Result<T,E> and Option<T> builtin methods:
//! - unwrap()
//! - unwrap_or()
//! - is_some() / is_none()
//! - is_ok() / is_err()
//! - map()
//! - and_then()

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

/// P2-6 Test 1: Option::Some.unwrap() returns the inner value.
#[test]
fn test_option_some_unwrap() {
    let src = r#"
fn make_some() -> Option<i64> {
    Some(99)
}
fn main() -> i64 {
    let opt = make_some();
    opt.unwrap()
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "option unwrap failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(99)));
}

/// P2-6 Test 2: Option::None.unwrap_or(default) returns default.
#[test]
fn test_option_none_unwrap_or() {
    let src = r#"
fn get_none() -> Option<i64> {
    None
}
fn main() -> i64 {
    let opt = get_none();
    opt.unwrap_or(42)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "none unwrap_or failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(42)));
}

/// P2-6 Test 3: Option is_some / is_none.
#[test]
fn test_option_is_some_is_none() {
    let src = r#"
fn main() -> i64 {
    let some_val = Some(5);
    let none_val: Option<i64> = None;
    if some_val.is_some() && none_val.is_none() {
        1
    } else {
        0
    }
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "is_some/is_none failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(1)));
}

/// P2-6 Test 4: Result::Ok.unwrap() returns inner value.
#[test]
fn test_result_ok_unwrap() {
    let src = r#"
fn make_ok() -> Result<i64, str> {
    Ok(77)
}
fn main() -> i64 {
    let r = make_ok();
    r.unwrap()
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "result ok unwrap failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(77)));
}

/// P2-6 Test 5: Result is_ok / is_err.
#[test]
fn test_result_is_ok_is_err() {
    let src = r#"
fn main() -> i64 {
    let ok_val = Ok(5);
    let err_val = Err("oops");
    if ok_val.is_ok() && err_val.is_err() {
        1
    } else {
        0
    }
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "is_ok/is_err failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(1)));
}

/// P2-6 Test 6: Option::Some.map() transforms inner value.
#[test]
fn test_option_map_some() {
    let src = r#"
fn double(x: i64) -> i64 { x * 2 }
fn main() -> i64 {
    let opt = Some(21);
    let doubled = opt.map(double);
    doubled.unwrap()
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "option map failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(42)));
}

/// P2-6 Test 7: Option::None.map() returns None (not evaluated).
#[test]
fn test_option_none_map_returns_none() {
    let src = r#"
fn double(x: i64) -> i64 { x * 2 }
fn main() -> i64 {
    let opt: Option<i64> = None;
    let mapped = opt.map(double);
    if mapped.is_none() { 1 } else { 0 }
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "option none map failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(1)));
}

/// P2-6 Test 8: Result::Ok.unwrap_or() returns inner value.
#[test]
fn test_result_ok_unwrap_or() {
    let src = r#"
fn main() -> i64 {
    let r = Ok(55);
    r.unwrap_or(0)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "result ok unwrap_or failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(55)));
}
