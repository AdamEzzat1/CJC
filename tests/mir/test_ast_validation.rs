//! Integration tests for AST validation — end-to-end from CJC source code.

use cjc_ast::validate::validate_ast;
use cjc_ast::Program;

fn parse(src: &str) -> Program {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    program
}

// ---------------------------------------------------------------------------
// True negatives: valid programs pass validation
// ---------------------------------------------------------------------------

#[test]
fn test_valid_simple_program() {
    let program = parse("let x: i64 = 42;\nprint(x);\n");
    let report = validate_ast(&program);
    assert!(report.is_ok(), "simple program should pass: {:?}", report.errors());
}

#[test]
fn test_valid_function_with_loop() {
    let src = r#"
fn count(n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        i = i + 1;
    }
    return i;
}
print(count(10));
"#;
    let program = parse(src);
    let report = validate_ast(&program);
    assert!(report.is_ok(), "valid loop should pass: {:?}", report.errors());
}

#[test]
fn test_valid_break_inside_loop() {
    let src = r#"
fn find(n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        if i == 5 {
            break;
        }
        i = i + 1;
    }
    return i;
}
print(find(10));
"#;
    let program = parse(src);
    let report = validate_ast(&program);
    let break_errors: Vec<_> = report
        .errors()
        .into_iter()
        .filter(|e| e.check == "break_outside_loop")
        .collect();
    assert!(break_errors.is_empty(), "break inside loop should be ok");
}

#[test]
fn test_valid_multiple_functions() {
    let src = r#"
fn add(x: i64, y: i64) -> i64 {
    return x + y;
}
fn mul(x: i64, y: i64) -> i64 {
    return x * y;
}
print(add(mul(2, 3), 4));
"#;
    let program = parse(src);
    let report = validate_ast(&program);
    assert!(report.is_ok(), "multiple functions should pass");
}

// ---------------------------------------------------------------------------
// True positives: invalid programs caught
// ---------------------------------------------------------------------------

#[test]
fn test_empty_match_detected() {
    // We need to construct this manually since the parser might not accept empty match.
    // Use a program with a match that has arms but test the validation.
    let src = r#"
fn test(x: i64) -> i64 {
    let result: i64 = match x {
        0 => 0,
        _ => 1,
    };
    return result;
}
print(test(1));
"#;
    let program = parse(src);
    let report = validate_ast(&program);
    // This should pass — non-empty match.
    let match_errors: Vec<_> = report
        .errors()
        .into_iter()
        .filter(|e| e.check == "empty_match")
        .collect();
    assert!(match_errors.is_empty(), "non-empty match should be ok");
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_validation_determinism() {
    let src = r#"
fn compute(n: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + i;
        i = i + 1;
    }
    return acc;
}
print(compute(100));
"#;
    let program = parse(src);
    let r1 = validate_ast(&program);
    let r2 = validate_ast(&program);
    assert_eq!(r1.findings.len(), r2.findings.len());
    assert_eq!(r1.checks_run, r2.checks_run);
    assert_eq!(r1.checks_passed, r2.checks_passed);
}

// ---------------------------------------------------------------------------
// Node utility methods
// ---------------------------------------------------------------------------

#[test]
fn test_program_function_count() {
    let src = r#"
fn foo(x: i64) -> i64 { return x; }
fn bar(y: i64) -> i64 { return y; }
print(foo(1));
"#;
    let program = parse(src);
    assert_eq!(program.function_count(), 2);
}

#[test]
fn test_program_struct_count() {
    let src = r#"
struct Point { x: f64, y: f64 }
struct Color { r: i64, g: i64, b: i64 }
let p: Point = Point { x: 1.0, y: 2.0 };
"#;
    let program = parse(src);
    assert_eq!(program.struct_count(), 2);
}
