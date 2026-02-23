//! Phase 2 Audit Tests: P1-2 Tail Call Optimization
//!
//! Tests for the interpreter-level tail call trampoline:
//! - Tail-recursive functions don't overflow the Rust stack
//! - Results are correct
//! - Non-tail recursion still works correctly

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

/// P1-2 Test 1: Tail-recursive countdown — returns zero without stack overflow.
/// Uses 100,000 iterations to stress-test the trampoline.
#[test]
fn test_tco_countdown_large() {
    let src = r#"
fn countdown(n: i64) -> i64 {
    if n == 0 {
        return 0;
    }
    return countdown(n - 1);
}
fn main() -> i64 {
    countdown(100000)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "TCO countdown failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(0)));
}

/// P1-2 Test 2: Tail-recursive sum accumulator (classic TCO example).
#[test]
fn test_tco_sum_accumulator() {
    let src = r#"
fn sum_tail(n: i64, acc: i64) -> i64 {
    if n == 0 {
        return acc;
    }
    return sum_tail(n - 1, acc + n);
}
fn main() -> i64 {
    sum_tail(1000, 0)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "TCO sum failed: {:?}", result);
    // sum(1..=1000) = 500500
    assert!(matches!(result.unwrap(), Value::Int(500500)));
}

/// P1-2 Test 3: Non-tail recursion (fibonacci) still produces correct results.
/// fib(10) = 55.
#[test]
fn test_non_tco_fibonacci_correct() {
    let src = r#"
fn fib(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    fib(n - 1) + fib(n - 2)
}
fn main() -> i64 {
    fib(10)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "fib(10) failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(55)));
}

/// P1-2 Test 4: Tail call with multiple arguments.
#[test]
fn test_tco_multiple_args() {
    let src = r#"
fn multiply_tail(n: i64, m: i64, acc: i64) -> i64 {
    if n == 0 {
        return acc;
    }
    return multiply_tail(n - 1, m, acc + m);
}
fn main() -> i64 {
    multiply_tail(100, 7, 0)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "TCO multi-arg failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(700)));
}

/// P1-2 Test 5: Result-position tail call (body result, not return stmt).
#[test]
fn test_tco_result_position() {
    let src = r#"
fn count_down_expr(n: i64) -> i64 {
    if n == 0 { 0 } else { count_down_expr(n - 1) }
}
fn main() -> i64 {
    count_down_expr(50000)
}
"#;
    let result = run_src(src);
    assert!(result.is_ok(), "TCO result-position failed: {:?}", result);
    assert!(matches!(result.unwrap(), Value::Int(0)));
}
