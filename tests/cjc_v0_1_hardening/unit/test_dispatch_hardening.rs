//! Dispatch layer hardening tests — operator dispatch via CJC programs.
//! Since operator dispatch is internal to the executors, we test it through
//! CJC program execution and verify correct results.

#[path = "../helpers.rs"]
mod helpers;
use helpers::*;

// ============================================================
// Integer arithmetic dispatch
// ============================================================

#[test]
fn dispatch_int_add() {
    let out = run_mir("fn main() { print(3 + 4); }");
    assert_eq!(out, vec!["7"]);
}

#[test]
fn dispatch_int_sub() {
    let out = run_mir("fn main() { print(10 - 3); }");
    assert_eq!(out, vec!["7"]);
}

#[test]
fn dispatch_int_mul() {
    let out = run_mir("fn main() { print(6 * 7); }");
    assert_eq!(out, vec!["42"]);
}

#[test]
fn dispatch_int_div() {
    let out = run_mir("fn main() { print(20 / 4); }");
    assert_eq!(out, vec!["5"]);
}

#[test]
fn dispatch_int_mod() {
    let out = run_mir("fn main() { print(17 % 5); }");
    assert_eq!(out, vec!["2"]);
}

// ============================================================
// Float arithmetic dispatch
// ============================================================

#[test]
fn dispatch_float_add() {
    let out = run_mir("fn main() { print(1.5 + 2.5); }");
    assert_eq!(out, vec!["4"]);
}

#[test]
fn dispatch_float_mul() {
    let out = run_mir("fn main() { print(3.0 * 2.0); }");
    assert_eq!(out, vec!["6"]);
}

#[test]
fn dispatch_float_div() {
    let out = run_mir("fn main() { print(10.0 / 4.0); }");
    assert_eq!(out, vec!["2.5"]);
}

// ============================================================
// Comparison dispatch
// ============================================================

#[test]
fn dispatch_lt() {
    let out = run_mir("fn main() { print(1 < 2); }");
    assert_eq!(out, vec!["true"]);
}

#[test]
fn dispatch_gt() {
    let out = run_mir("fn main() { print(2 > 1); }");
    assert_eq!(out, vec!["true"]);
}

#[test]
fn dispatch_eq() {
    let out = run_mir("fn main() { print(42 == 42); }");
    assert_eq!(out, vec!["true"]);
}

#[test]
fn dispatch_ne() {
    let out = run_mir("fn main() { print(1 != 2); }");
    assert_eq!(out, vec!["true"]);
}

#[test]
fn dispatch_le() {
    let out = run_mir("fn main() { print(1 <= 1); }");
    assert_eq!(out, vec!["true"]);
}

#[test]
fn dispatch_ge() {
    let out = run_mir("fn main() { print(2 >= 2); }");
    assert_eq!(out, vec!["true"]);
}

// ============================================================
// Unary dispatch
// ============================================================

#[test]
fn dispatch_unary_neg_int() {
    let out = run_mir("fn main() { print(-42); }");
    assert_eq!(out, vec!["-42"]);
}

#[test]
fn dispatch_unary_neg_float() {
    let out = run_mir("fn main() { print(-3.14); }");
    let val: f64 = out[0].parse().unwrap();
    assert!((val + 3.14).abs() < 1e-10);
}

#[test]
fn dispatch_unary_not() {
    let out = run_mir("fn main() { print(!true); print(!false); }");
    assert_eq!(out, vec!["false", "true"]);
}

// ============================================================
// String operations
// ============================================================

#[test]
fn dispatch_string_concat() {
    let out = run_mir(r#"fn main() { print("hello" + " " + "world"); }"#);
    assert_eq!(out, vec!["hello world"]);
}

// ============================================================
// Mixed-type operations
// ============================================================

#[test]
fn dispatch_int_float_promotion() {
    // int + float may promote to float
    let out = run_mir("fn main() { print(1 + 2.0); }");
    let val: f64 = out[0].parse().unwrap();
    assert!((val - 3.0).abs() < 1e-10);
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn dispatch_zero_division_float() {
    let out = run_mir("fn main() { print(1.0 / 0.0); }");
    // Should produce Inf or error, not panic
    assert!(!out.is_empty(), "Float division by zero should produce output");
}

#[test]
fn dispatch_negative_modulo() {
    let out = run_mir("fn main() { print(-7 % 3); }");
    // Result depends on semantics (C-style vs math-style)
    assert!(!out.is_empty(), "Negative modulo should produce output");
}

#[test]
fn dispatch_boolean_and_or() {
    let out = run_mir(r#"
fn main() {
    print(true && false);
    print(true || false);
    print(false && false);
    print(false || true);
}
"#);
    assert_eq!(out, vec!["false", "true", "false", "true"]);
}
