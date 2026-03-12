//! Type checker hardening tests — inference, unification, error reporting.

/// Integer literal has type i64.
#[test]
fn type_int_literal() {
    let src = "fn main() -> i64 { 42 }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    // Should execute without type error
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok(), "Int literal should type-check: {:?}", result.err());
}

/// Float literal has type f64.
#[test]
fn type_float_literal() {
    let src = "fn main() -> f64 { 3.14 }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok(), "Float literal should type-check: {:?}", result.err());
}

/// Bool literal has type bool.
#[test]
fn type_bool_literal() {
    let src = "fn main() -> bool { true }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok(), "Bool literal should type-check: {:?}", result.err());
}

/// String literal has type str.
#[test]
fn type_string_literal() {
    let src = r#"fn main() -> str { "hello" }"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok(), "String literal should type-check: {:?}", result.err());
}

/// Binary ops with matching types.
#[test]
fn type_binary_op_int() {
    let src = "fn main() -> i64 { 1 + 2 }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
}

/// Binary ops with float arithmetic.
#[test]
fn type_binary_op_float() {
    let src = "fn main() -> f64 { 1.0 + 2.5 }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
}

/// Comparison operators return bool.
#[test]
fn type_comparison_bool() {
    let src = "fn main() -> bool { 1 < 2 }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
}

/// Let binding infers type from RHS.
#[test]
fn type_let_inference() {
    let src = r#"
fn main() -> i64 {
    let x = 42;
    x
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
}

/// Function call with correct argument types.
#[test]
fn type_function_call() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(3, 4) }
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
}

/// Recursive function.
#[test]
fn type_recursive_function() {
    let src = r#"
fn factorial(n: i64) -> i64 {
    if n <= 1 { return 1; }
    n * factorial(n - 1)
}
fn main() -> i64 { factorial(5) }
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
}

/// Struct field access.
#[test]
fn type_struct_field_access() {
    let src = r#"
struct Point { x: f64, y: f64 }
fn main() -> f64 {
    let p = Point { x: 1.0, y: 2.0 };
    p.x
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
}

/// Closure type-checks correctly (tested via MIR — eval v1 has closure limitations).
#[test]
fn type_closure_capture() {
    let src = r#"
fn main() -> i64 {
    let f = |x: i64| x * 2;
    f(5)
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    // Use MIR executor since eval v1 can't call closure variables
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    assert!(result.is_ok(), "Closure should type-check and execute: {:?}", result.err());
}

/// Void function (no return type annotation).
#[test]
fn type_void_function() {
    let src = r#"
fn greet() {
    print("hello");
}
fn main() -> i64 {
    greet();
    0
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
}
