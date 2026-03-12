//! Parser hardening tests — edge cases, error recovery, and boundary conditions.

/// Empty program parses without panic.
#[test]
fn parse_empty_program() {
    let (program, diags) = cjc_parser::parse_source("");
    assert!(program.declarations.is_empty() || !diags.has_errors());
}

/// Single function declaration.
#[test]
fn parse_minimal_function() {
    let src = "fn main() -> i64 { 42 }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Minimal function should parse: {:?}", diags.diagnostics);
    assert!(!program.declarations.is_empty());
}

/// Function with typed parameters.
#[test]
fn parse_typed_params() {
    let src = "fn add(x: i64, y: i64) -> i64 { x + y }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Typed params should parse: {:?}", diags.diagnostics);
}

/// Let binding with type annotation.
#[test]
fn parse_let_with_type() {
    let src = "fn main() -> i64 { let x: i64 = 42; x }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Let with type should parse: {:?}", diags.diagnostics);
}

/// If-else as expression.
#[test]
fn parse_if_expression() {
    let src = r#"
fn main() -> i64 {
    let x = if true { 1 } else { 2 };
    x
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "If-expression should parse: {:?}", diags.diagnostics);
}

/// Nested if-else chains.
#[test]
fn parse_nested_if_else() {
    let src = r#"
fn main() -> i64 {
    let x = if true { 1 } else if false { 2 } else { 3 };
    x
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Nested if-else should parse: {:?}", diags.diagnostics);
}

/// While loop.
#[test]
fn parse_while_loop() {
    let src = r#"
fn main() -> i64 {
    let mut i: i64 = 0;
    while i < 10 {
        i = i + 1;
    }
    i
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "While loop should parse: {:?}", diags.diagnostics);
}

/// For-in loop.
#[test]
fn parse_for_loop() {
    let src = r#"
fn main() -> i64 {
    let mut sum: i64 = 0;
    for i in 0..10 {
        sum = sum + i;
    }
    sum
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "For-in loop should parse: {:?}", diags.diagnostics);
}

/// Match expression.
#[test]
fn parse_match_expression() {
    let src = r#"
fn main() -> i64 {
    let x: i64 = 5;
    match x {
        1 => 10,
        2 => 20,
        _ => 0,
    }
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Match expression should parse: {:?}", diags.diagnostics);
}

/// Struct definition.
#[test]
fn parse_struct_def() {
    let src = r#"
struct Point {
    x: f64,
    y: f64,
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Struct definition should parse: {:?}", diags.diagnostics);
}

/// Enum definition.
#[test]
fn parse_enum_def() {
    let src = r#"
enum Color {
    Red,
    Green,
    Blue,
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Enum definition should parse: {:?}", diags.diagnostics);
}

/// Lambda expression.
#[test]
fn parse_lambda() {
    let src = r#"
fn main() -> i64 {
    let f = |x: i64| x + 1;
    f(5)
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Lambda should parse: {:?}", diags.diagnostics);
}

/// Decorator syntax.
#[test]
fn parse_decorator() {
    let src = r#"
@timed
fn expensive() -> i64 { 42 }
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Decorator should parse: {:?}", diags.diagnostics);
}

/// Default parameter.
#[test]
fn parse_default_param() {
    let src = "fn solve(x: f64, tol: f64 = 1e-6) -> f64 { x }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Default param should parse: {:?}", diags.diagnostics);
}

/// Array literal.
#[test]
fn parse_array_literal() {
    let src = "fn main() -> Any { [1, 2, 3] }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Array literal should parse: {:?}", diags.diagnostics);
}

/// Tuple literal.
#[test]
fn parse_tuple_literal() {
    let src = "fn main() -> Any { (1, 2.0, true) }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Tuple literal should parse: {:?}", diags.diagnostics);
}

/// Complex binary expression with operator precedence.
#[test]
fn parse_operator_precedence() {
    let src = "fn main() -> i64 { 1 + 2 * 3 - 4 / 2 }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Operator precedence should parse: {:?}", diags.diagnostics);
}

/// Missing closing brace should produce error, not panic.
#[test]
fn parse_missing_closing_brace() {
    let src = "fn main() -> i64 { 42";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(diags.has_errors(), "Missing brace should produce error");
}

/// Missing function body.
#[test]
fn parse_missing_function_body() {
    let src = "fn main() -> i64";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(diags.has_errors(), "Missing function body should produce error");
}

/// Unexpected token should produce error, not panic.
#[test]
fn parse_unexpected_token() {
    let src = "fn 123invalid() -> i64 { 42 }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(diags.has_errors(), "Unexpected token should produce error");
}

/// Deeply nested expressions don't cause stack overflow in the parser.
#[test]
fn parse_deeply_nested_binary() {
    let depth = 100;
    let src = format!(
        "fn main() -> i64 {{ {} }}",
        (0..depth).map(|_| "1 + ").collect::<String>() + "1"
    );
    let (_, diags) = cjc_parser::parse_source(&src);
    assert!(!diags.has_errors(), "Deep binary should parse: {:?}", diags.diagnostics);
}

/// Multiple sequential function declarations.
#[test]
fn parse_multiple_functions() {
    let src = r#"
fn a() -> i64 { 1 }
fn b() -> i64 { 2 }
fn c() -> i64 { 3 }
fn main() -> i64 { a() + b() + c() }
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Multiple functions should parse: {:?}", diags.diagnostics);
    assert!(program.declarations.len() >= 4, "Should have at least 4 declarations");
}

/// String interpolation (f-string).
#[test]
fn parse_fstring() {
    let src = r#"fn main() -> str { f"hello {42}" }"#;
    let (_, diags) = cjc_parser::parse_source(src);
    // f-strings may or may not be supported; should not panic
    let _ = diags;
}

/// Return statement.
#[test]
fn parse_return_statement() {
    let src = "fn main() -> i64 { return 42; }";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Return statement should parse: {:?}", diags.diagnostics);
}

/// Break and continue in loops.
#[test]
fn parse_break_continue() {
    let src = r#"
fn main() -> i64 {
    let mut i: i64 = 0;
    while true {
        if i >= 10 {
            break;
        }
        i = i + 1;
        continue;
    }
    i
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Break/continue should parse: {:?}", diags.diagnostics);
}

/// Method call syntax.
#[test]
fn parse_method_call() {
    let src = r#"
fn main() -> Any {
    let t = Tensor.zeros([2, 3]);
    t.shape()
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Method call should parse: {:?}", diags.diagnostics);
}
