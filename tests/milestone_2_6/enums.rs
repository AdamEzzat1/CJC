// Milestone 2.6 — Enum Tests
//
// Tests for enum declaration, construction, pattern matching via the
// tree-walk evaluator (end-to-end: source -> lex -> parse -> eval).
//
// CJC enum variant syntax:
//   Construction:  bare variant name, e.g. `Red`, `Circle(3.14)`, `Some(42)`
//   Match pattern: `Some(x) =>`, `Circle(r) =>` (payload variants)
//   Unit variants in match: use `_ =>` or binding since bare `Red =>` acts as a binding
//
// Covers:
//   - Simple C-like enums (unit variants)
//   - Enums with payload fields
//   - Variant construction via call syntax
//   - Match with variant patterns and nested bindings
//   - Prelude enum variants (Some, None, Ok, Err)

use cjc_runtime::Value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_eval(source: &str) -> cjc_eval::EvalResult {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program)
}

fn run_eval_output(source: &str) -> Vec<String> {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, _) = lexer.tokenize();
    let parser = cjc_parser::Parser::new(tokens);
    let (program, _) = parser.parse_program();
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output
}

// ---------------------------------------------------------------------------
// Tests: Unit variant construction
// ---------------------------------------------------------------------------

#[test]
fn enum_unit_variant_construction() {
    // Bare variant name for a user-defined enum
    let src = r#"
        enum Color { Red, Green, Blue }
        fn main() {
            Red
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Color");
            assert_eq!(variant, "Red");
            assert!(fields.is_empty());
        }
        other => panic!("expected Enum Color::Red, got {:?}", other),
    }
}

#[test]
fn enum_unit_variant_green() {
    let src = r#"
        enum Color { Red, Green, Blue }
        fn main() {
            Green
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Color");
            assert_eq!(variant, "Green");
            assert!(fields.is_empty());
        }
        other => panic!("expected Enum Color::Green, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Tests: Payload variant construction
// ---------------------------------------------------------------------------

#[test]
fn enum_payload_single_field() {
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() {
            Circle(3.14)
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Shape");
            assert_eq!(variant, "Circle");
            assert_eq!(fields.len(), 1);
            match &fields[0] {
                Value::Float(v) => assert!((v - 3.14).abs() < 1e-12),
                other => panic!("expected Float field, got {:?}", other),
            }
        }
        other => panic!("expected Enum, got {:?}", other),
    }
}

#[test]
fn enum_payload_two_fields() {
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() {
            Rect(10.0, 20.0)
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Shape");
            assert_eq!(variant, "Rect");
            assert_eq!(fields.len(), 2);
            match (&fields[0], &fields[1]) {
                (Value::Float(w), Value::Float(h)) => {
                    assert!((w - 10.0).abs() < 1e-12);
                    assert!((h - 20.0).abs() < 1e-12);
                }
                other => panic!("expected (Float, Float), got {:?}", other),
            }
        }
        other => panic!("expected Enum, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Tests: Match with payload variant patterns
// ---------------------------------------------------------------------------

#[test]
fn enum_match_payload_with_binding() {
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() -> f64 {
            let s = Circle(5.0);
            match s {
                Circle(r) => r,
                Rect(w, h) => w + h,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Float(v) => assert!((v - 5.0).abs() < 1e-12),
        other => panic!("expected Float(5.0), got {:?}", other),
    }
}

#[test]
fn enum_match_payload_second_arm() {
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() -> f64 {
            let s = Rect(3.0, 4.0);
            match s {
                Circle(r) => r,
                Rect(w, h) => w + h,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Float(v) => assert!((v - 7.0).abs() < 1e-12),
        other => panic!("expected Float(7.0), got {:?}", other),
    }
}

#[test]
fn enum_match_with_wildcard() {
    // Wildcard catches variants not explicitly matched
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn main() -> i64 {
            let s = Rect(3.0, 4.0);
            match s {
                Circle(r) => 1,
                _ => 99,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(99)));
}

// ---------------------------------------------------------------------------
// Tests: Prelude variant construction
// ---------------------------------------------------------------------------

#[test]
fn enum_some_construction() {
    let src = r#"
        fn main() {
            Some(42)
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "Some");
            assert_eq!(fields.len(), 1);
            assert!(matches!(fields[0], Value::Int(42)));
        }
        other => panic!("expected Enum Option::Some(42), got {:?}", other),
    }
}

#[test]
fn enum_none_construction() {
    let src = r#"
        fn main() {
            None
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    match result {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "None");
            assert!(fields.is_empty());
        }
        other => panic!("expected Enum Option::None, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Tests: Match output with function calls
// ---------------------------------------------------------------------------

#[test]
fn enum_match_prints_variant_value() {
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn area(s: Shape) -> f64 {
            match s {
                Circle(r) => r * r * 3.14159,
                Rect(w, h) => w * h,
            }
        }
        fn main() {
            print(area(Circle(2.0)));
            print(area(Rect(3.0, 4.0)));
        }
    "#;
    let output = run_eval_output(src);
    assert_eq!(output.len(), 2);
    let circle_area: f64 = output[0].parse().expect("parse float");
    assert!((circle_area - 12.56636).abs() < 0.001);
    let rect_area: f64 = output[1].parse().expect("parse float");
    assert!((rect_area - 12.0).abs() < 1e-12);
}

#[test]
fn enum_pass_as_function_argument() {
    let src = r#"
        enum Shape { Circle(f64), Rect(f64, f64) }
        fn describe(s: Shape) -> i64 {
            match s {
                Circle(r) => 1,
                Rect(w, h) => 2,
            }
        }
        fn main() -> i64 {
            describe(Circle(1.0)) + describe(Rect(2.0, 3.0))
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(3)));
}

#[test]
fn enum_return_from_function() {
    let src = r#"
        enum Maybe { Just(i64), Nothing }
        fn safe_div(a: i64, b: i64) -> Maybe {
            if b == 0 {
                Nothing
            } else {
                Just(a / b)
            }
        }
        fn main() -> i64 {
            let r = safe_div(10, 2);
            match r {
                Just(v) => v,
                _ => -1,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn enum_return_unit_variant_from_function() {
    let src = r#"
        enum Maybe { Just(i64), Nothing }
        fn safe_div(a: i64, b: i64) -> Maybe {
            if b == 0 {
                Nothing
            } else {
                Just(a / b)
            }
        }
        fn main() -> i64 {
            let r = safe_div(10, 0);
            match r {
                Just(v) => v,
                _ => -1,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(-1)));
}

#[test]
fn enum_display_formatting() {
    let src = r#"
        fn main() {
            let val = Some(99);
            print(val);
        }
    "#;
    let output = run_eval_output(src);
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "Some(99)");
}

#[test]
fn enum_match_option_some_arm() {
    let src = r#"
        fn main() -> i64 {
            let x = Some(10);
            match x {
                Some(v) => v + 1,
                None => 0,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(11)));
}

#[test]
fn enum_match_option_none_arm() {
    let src = r#"
        fn main() -> i64 {
            let x = None;
            match x {
                Some(v) => v,
                None => -1,
            }
        }
    "#;
    let result = run_eval(src).expect("eval failed");
    assert!(matches!(result, Value::Int(-1)));
}
