//! LH-10: Records / Value Semantics
//!
//! Tests for the `record` keyword — immutable value types with structural
//! equality.  Records are like structs but field assignment is a type error.

use cjc_types::TypeChecker;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    for d in &diags.diagnostics {
        eprintln!("  parse diag: {d:?}");
    }
    assert!(!diags.has_errors(), "unexpected parse errors");
    program
}

fn eval_str(src: &str) -> String {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let val = interp.exec(&program).expect("eval failed");
    format!("{val}")
}

fn mir_str(src: &str) -> String {
    let program = parse(src);
    let (val, _exec) =
        cjc_mir_exec::run_program_with_executor(&program, 42).expect("MIR exec failed");
    format!("{val}")
}

// ---------------------------------------------------------------------------
// 1. Parsing — record keyword is recognized
// ---------------------------------------------------------------------------

#[test]
fn test_record_parses() {
    let result = eval_str(r#"
record Point {
    x: f64,
    y: f64
}
fn main() -> f64 {
    let p = Point { x: 1.0, y: 2.0 };
    p.x + p.y
}
"#);
    assert_eq!(result, "3");
}

#[test]
fn test_record_parses_mir() {
    let result = mir_str(r#"
record Point {
    x: f64,
    y: f64
}
fn main() -> f64 {
    let p = Point { x: 1.0, y: 2.0 };
    p.x + p.y
}
"#);
    assert_eq!(result, "3");
}

// ---------------------------------------------------------------------------
// 2. Parity — eval and MIR produce identical results
// ---------------------------------------------------------------------------

#[test]
fn test_record_parity_simple() {
    let src = r#"
record Vec2 {
    x: f64,
    y: f64
}
fn main() -> f64 {
    let v = Vec2 { x: 3.0, y: 4.0 };
    v.x * v.x + v.y * v.y
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity: eval={ev} mir={mv}");
    assert_eq!(ev, "25");
}

#[test]
fn test_record_parity_nested() {
    let src = r#"
record Point {
    x: f64,
    y: f64
}
record Line {
    a: Any,
    b: Any
}
fn main() -> f64 {
    let p1 = Point { x: 1.0, y: 2.0 };
    let p2 = Point { x: 3.0, y: 4.0 };
    let line = Line { a: p1, b: p2 };
    line.b.x
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity: eval={ev} mir={mv}");
    assert_eq!(ev, "3");
}

// ---------------------------------------------------------------------------
// 3. Immutability — record field assignment rejected at type-check time
// ---------------------------------------------------------------------------

#[test]
fn test_record_immutability_type_error() {
    let src = r#"
record Color {
    r: i64,
    g: i64,
    b: i64
}
fn main() -> i64 {
    let mut c = Color { r: 255, g: 0, b: 0 };
    c.r = 128;
    c.r
}
"#;
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors(), "parse errors");

    // Type-check should emit E0160 (record field assignment)
    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    let has_e0160 = checker.diagnostics.diagnostics.iter()
        .any(|d| d.code == "E0160");
    assert!(
        has_e0160,
        "expected E0160 error for record field assignment, got: {:?}",
        checker.diagnostics.diagnostics
    );
}

// ---------------------------------------------------------------------------
// 4. Runtime immutability enforcement (belt-and-suspenders)
// ---------------------------------------------------------------------------

#[test]
fn test_record_runtime_immutability_eval() {
    let src = r#"
record Frozen {
    val: i64
}
fn main() -> i64 {
    let mut f = Frozen { val: 42 };
    f.val = 99;
    f.val
}
"#;
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "should reject record field assignment at runtime");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("immutable") || err_msg.contains("record"),
        "error should mention immutability: {err_msg}"
    );
}

#[test]
fn test_record_runtime_immutability_mir() {
    let src = r#"
record Frozen {
    val: i64
}
fn main() -> i64 {
    let mut f = Frozen { val: 42 };
    f.val = 99;
    f.val
}
"#;
    let program = parse(src);
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    assert!(result.is_err(), "should reject record field assignment at runtime");
    let err_msg = match result {
        Err(e) => format!("{e:?}"),
        Ok(_) => panic!("expected error"),
    };
    assert!(
        err_msg.contains("immutable") || err_msg.contains("record"),
        "error should mention immutability: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// 5. Structural equality
// ---------------------------------------------------------------------------

#[test]
fn test_record_structural_equality_eval() {
    let src = r#"
record Color {
    r: i64,
    g: i64,
    b: i64
}
fn check() -> i64 {
    let a = Color { r: 255, g: 128, b: 0 };
    let b = Color { r: 255, g: 128, b: 0 };
    let c = Color { r: 0, g: 0, b: 0 };
    let eq1 = a == b;
    let ne1 = a != c;
    let eq2 = a == c;
    if eq1 {
        if ne1 {
            if eq2 {
                return 0;
            } else {
                return 1;
            }
        }
    }
    0
}
fn main() -> i64 {
    check()
}
"#;
    let result = eval_str(src);
    assert_eq!(result, "1", "structural equality should work");
}

#[test]
fn test_record_structural_equality_mir() {
    let src = r#"
record Color {
    r: i64,
    g: i64,
    b: i64
}
fn check() -> i64 {
    let a = Color { r: 255, g: 128, b: 0 };
    let b = Color { r: 255, g: 128, b: 0 };
    let c = Color { r: 0, g: 0, b: 0 };
    let eq1 = a == b;
    let ne1 = a != c;
    let eq2 = a == c;
    if eq1 {
        if ne1 {
            if eq2 {
                return 0;
            } else {
                return 1;
            }
        }
    }
    0
}
fn main() -> i64 {
    check()
}
"#;
    let result = mir_str(src);
    assert_eq!(result, "1", "structural equality should work");
}

#[test]
fn test_struct_equality_also_works() {
    let src = r#"
struct Pair {
    x: i64,
    y: i64
}
fn check() -> i64 {
    let a = Pair { x: 1, y: 2 };
    let b = Pair { x: 1, y: 2 };
    if a == b {
        return 1;
    }
    0
}
fn main() -> i64 {
    check()
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "1");
}

// ---------------------------------------------------------------------------
// 6. Record as function argument
// ---------------------------------------------------------------------------

#[test]
fn test_record_field_access() {
    let src = r#"
record Config {
    width: i64,
    height: i64
}
fn area(c: Any) -> i64 {
    c.width * c.height
}
fn main() -> i64 {
    let cfg = Config { width: 800, height: 600 };
    area(cfg)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "480000");
}

// ---------------------------------------------------------------------------
// 7. Record in pattern matching
// ---------------------------------------------------------------------------

#[test]
fn test_record_match_destructure() {
    let src = r#"
record Pair {
    first: i64,
    second: i64
}
fn sum_pair(p: Any) -> i64 {
    match p {
        Pair { first: a, second: b } => a + b,
    }
}
fn main() -> i64 {
    let p = Pair { first: 10, second: 20 };
    sum_pair(p)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "30");
}

// ---------------------------------------------------------------------------
// 8. Struct field assignment still works (mutable)
// ---------------------------------------------------------------------------

#[test]
fn test_struct_still_mutable() {
    let src = r#"
struct MutablePoint {
    x: i64,
    y: i64
}
fn main() -> i64 {
    let mut p = MutablePoint { x: 10, y: 20 };
    p.x = 99;
    p.x
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "99");
}

// ---------------------------------------------------------------------------
// 9. Determinism
// ---------------------------------------------------------------------------

#[test]
fn test_record_deterministic() {
    let src = r#"
record RGB {
    r: i64,
    g: i64,
    b: i64
}
fn main() -> i64 {
    let c = RGB { r: 10, g: 20, b: 30 };
    c.r + c.g + c.b
}
"#;
    let v1 = eval_str(src);
    let v2 = eval_str(src);
    let m1 = mir_str(src);
    let m2 = mir_str(src);
    assert_eq!(v1, v2);
    assert_eq!(m1, m2);
    assert_eq!(v1, m1);
}

// ---------------------------------------------------------------------------
// 10. Different record types not equal
// ---------------------------------------------------------------------------

#[test]
fn test_record_different_names_not_equal() {
    let src = r#"
record A {
    x: i64
}
record B {
    x: i64
}
fn check() -> i64 {
    let a = A { x: 1 };
    let b = B { x: 1 };
    if a == b {
        return 1;
    }
    0
}
fn main() -> i64 {
    check()
}
"#;
    let result = eval_str(src);
    assert_eq!(result, "0");
}
