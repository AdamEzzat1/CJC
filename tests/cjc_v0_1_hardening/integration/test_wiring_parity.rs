//! Parity tests — eval and MIR-exec must produce identical output.
//! This is the most critical integration gate.

#[path = "../helpers.rs"]
mod helpers;
use helpers::*;

/// Helper to assert parity between eval and MIR-exec.
fn check_parity(label: &str, src: &str) {
    let eval_out = run_eval(src);
    let mir_out = run_mir(src);
    assert_eq!(
        eval_out, mir_out,
        "[PARITY FAILURE: {label}]\nEval: {eval_out:?}\nMIR:  {mir_out:?}"
    );
}

// ============================================================
// Basic parity
// ============================================================

#[test]
fn parity_integer_literal() {
    check_parity("int literal", "fn main() { print(42); }");
}

#[test]
fn parity_float_literal() {
    check_parity("float literal", "fn main() { print(3.14); }");
}

#[test]
fn parity_bool_literal() {
    check_parity("bool literal", "fn main() { print(true); print(false); }");
}

#[test]
fn parity_string_literal() {
    check_parity("string literal", r#"fn main() { print("hello"); }"#);
}

// ============================================================
// Arithmetic parity
// ============================================================

#[test]
fn parity_int_arithmetic() {
    check_parity("int arith", "fn main() { print(1 + 2); print(10 - 3); print(4 * 5); print(20 / 4); }");
}

#[test]
fn parity_float_arithmetic() {
    check_parity("float arith", "fn main() { print(1.5 + 2.5); print(3.0 * 2.0); }");
}

#[test]
fn parity_modulo() {
    check_parity("modulo", "fn main() { print(17 % 5); }");
}

#[test]
fn parity_comparison() {
    check_parity("comparison", "fn main() { print(1 < 2); print(2 > 1); print(1 == 1); print(1 != 2); }");
}

// ============================================================
// Control flow parity
// ============================================================

#[test]
fn parity_if_else() {
    check_parity("if-else", r#"
fn main() {
    if 1 < 2 { print("yes"); } else { print("no"); }
}
"#);
}

#[test]
fn parity_while_loop() {
    check_parity("while", r#"
fn main() {
    let mut i: i64 = 0;
    while i < 5 {
        i = i + 1;
    }
    print(i);
}
"#);
}

#[test]
fn parity_for_loop() {
    check_parity("for", r#"
fn main() {
    let mut sum: i64 = 0;
    for i in 0..10 {
        sum = sum + i;
    }
    print(sum);
}
"#);
}

#[test]
fn parity_break() {
    check_parity("break", r#"
fn main() {
    let mut i: i64 = 0;
    while true {
        if i >= 5 { break; }
        i = i + 1;
    }
    print(i);
}
"#);
}

// ============================================================
// Function parity
// ============================================================

#[test]
fn parity_function_call() {
    check_parity("fn call", r#"
fn double(x: i64) -> i64 { x * 2 }
fn main() { print(double(21)); }
"#);
}

#[test]
fn parity_recursive_function() {
    check_parity("recursion", r#"
fn fact(n: i64) -> i64 {
    if n <= 1 { return 1; }
    n * fact(n - 1)
}
fn main() { print(fact(10)); }
"#);
}

#[test]
fn parity_multiple_functions() {
    check_parity("multi-fn", r#"
fn a(x: i64) -> i64 { x + 1 }
fn b(x: i64) -> i64 { x * 2 }
fn main() { print(b(a(20))); }
"#);
}

// ============================================================
// Closure parity
// ============================================================

/// Closure call via MIR (eval does not support calling closures stored in variables from source).
#[test]
fn parity_closure() {
    let out = run_mir(r#"
fn main() {
    let f = |x: i64| x + 1;
    print(f(41));
}
"#);
    assert_eq!(out[0], "42", "closure should return 42");
}

/// Closure with capture via MIR.
#[test]
fn parity_closure_capture() {
    let out = run_mir(r#"
fn main() {
    let n: i64 = 100;
    let f = |x: i64| x + n;
    print(f(42));
}
"#);
    assert_eq!(out[0], "142", "closure capture should return 142");
}

// ============================================================
// Match parity
// ============================================================

#[test]
fn parity_match() {
    check_parity("match", r#"
fn main() {
    let x: i64 = 3;
    let r = match x {
        1 => 10,
        2 => 20,
        3 => 30,
        _ => 0,
    };
    print(r);
}
"#);
}

// ============================================================
// Struct parity
// ============================================================

#[test]
fn parity_struct() {
    check_parity("struct", r#"
struct Point { x: f64, y: f64 }
fn main() {
    let p = Point { x: 1.0, y: 2.0 };
    print(p.x);
    print(p.y);
}
"#);
}

// ============================================================
// If-as-expression parity
// ============================================================

#[test]
fn parity_if_expression() {
    check_parity("if-expr", r#"
fn main() {
    let x = if true { 42 } else { 0 };
    print(x);
}
"#);
}

// ============================================================
// Builtin parity
// ============================================================

#[test]
fn parity_math_builtins() {
    check_parity("math builtins", r#"
fn main() {
    print(abs(-5.0));
    print(sqrt(4.0));
    print(sin(0.0));
    print(cos(0.0));
}
"#);
}

#[test]
fn parity_tensor_zeros() {
    check_parity("tensor zeros", r#"
fn main() {
    let t = Tensor.zeros([3]);
    print(t);
}
"#);
}

#[test]
fn parity_tensor_from_vec() {
    check_parity("tensor from_vec", r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
    print(t);
}
"#);
}
