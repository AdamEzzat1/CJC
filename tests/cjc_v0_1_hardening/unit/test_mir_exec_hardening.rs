//! MIR executor hardening tests — same as eval tests but via MIR pipeline.

#[path = "../helpers.rs"]
mod helpers;
use helpers::*;

// ============================================================
// Basic evaluation via MIR
// ============================================================

#[test]
fn mir_integer_arithmetic() {
    let out = run_mir("fn main() { print(1 + 2); print(10 - 3); print(4 * 5); print(20 / 4); }");
    assert_eq!(out, vec!["3", "7", "20", "5"]);
}

#[test]
fn mir_float_arithmetic() {
    let out = run_mir("fn main() { print(1.5 + 2.5); }");
    assert_eq!(out, vec!["4"]);
}

#[test]
fn mir_boolean_logic() {
    let out = run_mir("fn main() { print(true); print(false); }");
    assert_eq!(out, vec!["true", "false"]);
}

#[test]
fn mir_string_literal() {
    let out = run_mir(r#"fn main() { print("hello world"); }"#);
    assert_eq!(out, vec!["hello world"]);
}

// ============================================================
// Variable binding
// ============================================================

#[test]
fn mir_let_binding() {
    let out = run_mir("fn main() { let x: i64 = 42; print(x); }");
    assert_eq!(out, vec!["42"]);
}

#[test]
fn mir_let_mut_reassign() {
    let out = run_mir(r#"
fn main() {
    let mut x: i64 = 1;
    x = x + 1;
    print(x);
}
"#);
    assert_eq!(out, vec!["2"]);
}

// ============================================================
// Control flow
// ============================================================

#[test]
fn mir_if_else() {
    let out = run_mir(r#"
fn main() {
    if true { print("yes"); } else { print("no"); }
}
"#);
    assert_eq!(out, vec!["yes"]);
}

#[test]
fn mir_while_loop() {
    let out = run_mir(r#"
fn main() {
    let mut i: i64 = 0;
    while i < 5 {
        i = i + 1;
    }
    print(i);
}
"#);
    assert_eq!(out, vec!["5"]);
}

#[test]
fn mir_for_loop() {
    let out = run_mir(r#"
fn main() {
    let mut sum: i64 = 0;
    for i in 0..5 {
        sum = sum + i;
    }
    print(sum);
}
"#);
    assert_eq!(out, vec!["10"]);
}

#[test]
fn mir_break_in_loop() {
    let out = run_mir(r#"
fn main() {
    let mut i: i64 = 0;
    while true {
        if i >= 3 { break; }
        i = i + 1;
    }
    print(i);
}
"#);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn mir_continue_in_loop() {
    let out = run_mir(r#"
fn main() {
    let mut sum: i64 = 0;
    for i in 0..10 {
        if i % 2 == 0 { continue; }
        sum = sum + i;
    }
    print(sum);
}
"#);
    assert_eq!(out, vec!["25"]);
}

// ============================================================
// Functions
// ============================================================

#[test]
fn mir_function_call() {
    let out = run_mir(r#"
fn double(x: i64) -> i64 { x * 2 }
fn main() { print(double(21)); }
"#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn mir_recursive_factorial() {
    let out = run_mir(r#"
fn fact(n: i64) -> i64 {
    if n <= 1 { return 1; }
    n * fact(n - 1)
}
fn main() { print(fact(10)); }
"#);
    assert_eq!(out, vec!["3628800"]);
}

// ============================================================
// Closures
// ============================================================

#[test]
fn mir_closure_basic() {
    let out = run_mir(r#"
fn main() {
    let inc = |x: i64| x + 1;
    print(inc(41));
}
"#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn mir_closure_captures() {
    let out = run_mir(r#"
fn main() {
    let offset: i64 = 100;
    let add_offset = |x: i64| x + offset;
    print(add_offset(42));
}
"#);
    assert_eq!(out, vec!["142"]);
}

// ============================================================
// Match expressions
// ============================================================

#[test]
fn mir_match_int() {
    let out = run_mir(r#"
fn main() {
    let x: i64 = 2;
    let result = match x {
        1 => 10,
        2 => 20,
        _ => 0,
    };
    print(result);
}
"#);
    assert_eq!(out, vec!["20"]);
}

// ============================================================
// Structs
// ============================================================

#[test]
fn mir_struct_create_and_access() {
    let out = run_mir(r#"
struct Point { x: f64, y: f64 }
fn main() {
    let p = Point { x: 3.0, y: 4.0 };
    print(p.x);
    print(p.y);
}
"#);
    assert_eq!(out, vec!["3", "4"]);
}

// ============================================================
// Tensor operations
// ============================================================

#[test]
fn mir_tensor_zeros() {
    let out = run_mir(r#"
fn main() {
    let t = Tensor.zeros([3]);
    print(t);
}
"#);
    assert!(!out.is_empty(), "Tensor.zeros should produce output");
}

#[test]
fn mir_tensor_ones() {
    let out = run_mir(r#"
fn main() {
    let t = Tensor.ones([2, 2]);
    print(t);
}
"#);
    assert!(!out.is_empty(), "Tensor.ones should produce output");
}

#[test]
fn mir_tensor_from_vec() {
    let out = run_mir(r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
    print(t);
}
"#);
    assert!(!out.is_empty(), "Tensor.from_vec should produce output");
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn mir_empty_function() {
    let out = run_mir("fn main() { }");
    assert!(out.is_empty());
}

#[test]
fn mir_negative_literal() {
    let out = run_mir("fn main() { print(-42); }");
    assert_eq!(out, vec!["-42"]);
}

#[test]
fn mir_nested_function_calls() {
    let out = run_mir(r#"
fn inc(x: i64) -> i64 { x + 1 }
fn double(x: i64) -> i64 { x * 2 }
fn main() { print(double(inc(20))); }
"#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn mir_modulo_operator() {
    let out = run_mir("fn main() { print(17 % 5); }");
    assert_eq!(out, vec!["2"]);
}

#[test]
fn mir_multiple_print_calls() {
    let out = run_mir(r#"
fn main() {
    print(1);
    print(2);
    print(3);
}
"#);
    assert_eq!(out, vec!["1", "2", "3"]);
}

#[test]
fn mir_comparison_operators() {
    let out = run_mir(r#"
fn main() {
    print(1 < 2);
    print(2 > 1);
    print(1 == 1);
    print(1 != 2);
    print(1 <= 1);
    print(2 >= 2);
}
"#);
    assert_eq!(out, vec!["true", "true", "true", "true", "true", "true"]);
}
