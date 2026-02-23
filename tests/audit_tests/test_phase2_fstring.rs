//! Phase 2 Audit Tests: P2-5 String Interpolation (f"...")
//!
//! Tests for format string literals:
//! - Basic variable interpolation
//! - Multiple interpolations
//! - Expressions inside braces
//! - Empty interpolation holes
//! - Literal braces via {{ and }}

use cjc_parser::parse_source;
use cjc_mir_exec::run_program_with_executor;
use cjc_runtime::Value;
use std::rc::Rc;

fn run_src(src: &str) -> Result<(Value, Vec<String>), String> {
    let (prog, diags) = parse_source(src);
    if diags.has_errors() {

        return Err(format!("parse errors"));
    }
    run_program_with_executor(&prog, 42)
        .map(|(v, exec)| (v, exec.output.clone()))
        .map_err(|e| format!("{e}"))
}

/// P2-5 Test 1: Basic variable interpolation.
#[test]
fn test_fstring_basic_var() {
    let src = r#"
fn main() -> i64 {
    let name: str = "world";
    print(f"hello {name}!");
    0
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "fstring basic failed: {:?}", result);
    let (_, output) = result.unwrap();
    assert!(output.iter().any(|l| l.contains("hello world!")),
        "Expected 'hello world!' in output, got: {:?}", output);
}

/// P2-5 Test 2: Integer interpolation.
#[test]
fn test_fstring_int_interpolation() {
    let src = r#"
fn main() -> i64 {
    let x: i64 = 42;
    print(f"answer is {x}");
    0
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "fstring int failed: {:?}", result);
    let (_, output) = result.unwrap();
    assert!(output.iter().any(|l| l.contains("answer is 42")),
        "Expected 'answer is 42' in output, got: {:?}", output);
}

/// P2-5 Test 3: Multiple interpolations in one string.
#[test]
fn test_fstring_multiple_interp() {
    let src = r#"
fn main() -> i64 {
    let a: i64 = 1;
    let b: i64 = 2;
    print(f"{a} + {b} = {a + b}");
    0
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "fstring multiple failed: {:?}", result);
    let (_, output) = result.unwrap();
    assert!(output.iter().any(|l| l.contains("1 + 2 = 3")),
        "Expected '1 + 2 = 3' in output, got: {:?}", output);
}

/// P2-5 Test 4: Expression interpolation.
#[test]
fn test_fstring_expr_interpolation() {
    let src = r#"
fn main() -> i64 {
    let n: i64 = 5;
    print(f"n squared is {n * n}");
    0
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "fstring expr failed: {:?}", result);
    let (_, output) = result.unwrap();
    assert!(output.iter().any(|l| l.contains("n squared is 25")),
        "Expected 'n squared is 25' in output, got: {:?}", output);
}

/// P2-5 Test 5: fstring with no interpolations (just a string literal).
#[test]
fn test_fstring_no_interp() {
    let src = r#"
fn main() -> i64 {
    print(f"no interpolation here");
    0
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "fstring no-interp failed: {:?}", result);
    let (_, output) = result.unwrap();
    assert!(output.iter().any(|l| l.contains("no interpolation here")),
        "Expected literal string in output, got: {:?}", output);
}

/// P2-5 Test 6: fstring used as a value (assigned to variable).
#[test]
fn test_fstring_as_value() {
    let src = r#"
fn make_greeting(name: str) -> str {
    f"Hello, {name}!"
}
fn main() -> i64 {
    print(make_greeting("Alice"));
    0
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "fstring as value failed: {:?}", result);
    let (_, output) = result.unwrap();
    assert!(output.iter().any(|l| l.contains("Hello, Alice!")),
        "Expected 'Hello, Alice!' in output, got: {:?}", output);
}

/// P2-5 Test 7: fstring with boolean interpolation.
#[test]
fn test_fstring_bool_interpolation() {
    let src = r#"
fn main() -> i64 {
    let flag: bool = true;
    print(f"flag is {flag}");
    0
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "fstring bool failed: {:?}", result);
    let (_, output) = result.unwrap();
    assert!(output.iter().any(|l| l.contains("flag is true")),
        "Expected 'flag is true' in output, got: {:?}", output);
}
