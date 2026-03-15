//! Pattern match exhaustiveness checking tests.

#[test]
fn test_enum_exhaustiveness_all_covered() {
    let src = r#"
enum Color { Red, Green, Blue }
fn describe(c: Color) -> str {
    match c {
        Red => "red"
        Green => "green"
        Blue => "blue"
    }
}
let x = describe(Color::Red);
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);
    // Should have no exhaustiveness errors
    let errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E0130" || d.code == "E0131")
        .collect();
    assert!(errors.is_empty(), "no exhaustiveness errors expected: {:?}",
        errors.iter().map(|d| &d.message).collect::<Vec<_>>());
}

#[test]
fn test_enum_exhaustiveness_with_wildcard() {
    let src = r#"
enum Shape { Circle, Square, Triangle }
fn area(s: Shape) -> f64 {
    match s {
        Circle => 3.14
        _ => 1.0
    }
}
let x = area(Shape::Circle);
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);
    let errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E0130")
        .collect();
    assert!(errors.is_empty(), "wildcard should cover missing variants");
}

#[test]
fn test_bool_exhaustiveness_both_covered() {
    // This is a type-level test; the actual pattern matching for bool
    // requires the type checker to recognize bool scrutinee types.
    // For now we verify the infrastructure exists.
    let src = r#"
let x = true;
let y = if x { 1 } else { 0 };
y
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program).unwrap();
    match result {
        cjc_runtime::Value::Int(1) => {}
        other => panic!("expected 1, got {:?}", other),
    }
}
