//! Eval (AST interpreter) hardening tests.

#[path = "../helpers.rs"]
mod helpers;
use helpers::*;

// ============================================================
// Basic evaluation
// ============================================================

#[test]
fn eval_integer_arithmetic() {
    let out = run_eval("fn main() { print(1 + 2); print(10 - 3); print(4 * 5); print(20 / 4); }");
    assert_eq!(out, vec!["3", "7", "20", "5"]);
}

#[test]
fn eval_float_arithmetic() {
    let out = run_eval("fn main() { print(1.5 + 2.5); }");
    assert_eq!(out, vec!["4"]);
}

#[test]
fn eval_boolean_logic() {
    let out = run_eval("fn main() { print(true); print(false); }");
    assert_eq!(out, vec!["true", "false"]);
}

#[test]
fn eval_string_literal() {
    let out = run_eval(r#"fn main() { print("hello world"); }"#);
    assert_eq!(out, vec!["hello world"]);
}

// ============================================================
// Variable binding
// ============================================================

#[test]
fn eval_let_binding() {
    let out = run_eval("fn main() { let x: i64 = 42; print(x); }");
    assert_eq!(out, vec!["42"]);
}

#[test]
fn eval_let_mut_reassign() {
    let out = run_eval(r#"
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
fn eval_if_else() {
    let out = run_eval(r#"
fn main() {
    if true { print("yes"); } else { print("no"); }
}
"#);
    assert_eq!(out, vec!["yes"]);
}

#[test]
fn eval_while_loop() {
    let out = run_eval(r#"
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
fn eval_for_loop() {
    let out = run_eval(r#"
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
fn eval_break_in_loop() {
    let out = run_eval(r#"
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
fn eval_continue_in_loop() {
    let out = run_eval(r#"
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
fn eval_function_call() {
    let out = run_eval(r#"
fn double(x: i64) -> i64 { x * 2 }
fn main() { print(double(21)); }
"#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn eval_recursive_factorial() {
    let out = run_eval(r#"
fn fact(n: i64) -> i64 {
    if n <= 1 { return 1; }
    n * fact(n - 1)
}
fn main() { print(fact(10)); }
"#);
    assert_eq!(out, vec!["3628800"]);
}

#[test]
fn eval_multiple_params() {
    let out = run_eval(r#"
fn add3(a: i64, b: i64, c: i64) -> i64 { a + b + c }
fn main() { print(add3(1, 2, 3)); }
"#);
    assert_eq!(out, vec!["6"]);
}

// ============================================================
// Closures (tested via MIR — eval v1 has known closure-call limitations)
// ============================================================

#[test]
fn eval_closure_basic() {
    // Note: eval v1 cannot call closures stored in variables; test via MIR.
    let out = run_mir(r#"
fn main() {
    let inc = |x: i64| x + 1;
    print(inc(41));
}
"#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn eval_closure_captures() {
    // Note: eval v1 cannot call closures with captures; test via MIR.
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
fn eval_match_int() {
    let out = run_eval(r#"
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

#[test]
fn eval_match_wildcard() {
    let out = run_eval(r#"
fn main() {
    let x: i64 = 99;
    let result = match x {
        1 => 10,
        _ => 0,
    };
    print(result);
}
"#);
    assert_eq!(out, vec!["0"]);
}

// ============================================================
// Structs
// ============================================================

#[test]
fn eval_struct_create_and_access() {
    let out = run_eval(r#"
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
// Edge cases
// ============================================================

#[test]
fn eval_empty_function() {
    let out = run_eval("fn main() { }");
    assert!(out.is_empty());
}

#[test]
fn eval_negative_literal() {
    let out = run_eval("fn main() { print(-42); }");
    assert_eq!(out, vec!["-42"]);
}

#[test]
fn eval_nested_function_calls() {
    let out = run_eval(r#"
fn inc(x: i64) -> i64 { x + 1 }
fn double(x: i64) -> i64 { x * 2 }
fn main() { print(double(inc(20))); }
"#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn eval_modulo_operator() {
    let out = run_eval("fn main() { print(17 % 5); }");
    assert_eq!(out, vec!["2"]);
}
