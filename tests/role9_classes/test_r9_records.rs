//! Role 9, Requirement 1: Records as Default Data Carrier
//!
//! Records are immutable value types with structural equality — the preferred
//! way to carry data in CJC.  Tests verify:
//!   - Brace-literal construction and field access
//!   - Structural equality (==) and inequality (!=)
//!   - Compile-time immutability (E0160)
//!   - Runtime immutability (belt-and-suspenders in both eval and MIR)
//!   - Records as function arguments
//!   - Nested records
//!   - Pattern matching / destructuring
//!   - Struct contrast (mutable counterpart)
//!   - Parity between eval and MIR-exec
//!   - Determinism across runs

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
// 1. Basic record creation and field access
// ---------------------------------------------------------------------------

#[test]
fn test_record_creation_and_field_access() {
    let src = r#"
record Point {
    x: f64,
    y: f64
}
fn main() -> f64 {
    let p = Point { x: 3.0, y: 4.0 };
    p.x + p.y
}
"#;
    assert_eq!(eval_str(src), "7");
    assert_eq!(mir_str(src), "7");
}

// ---------------------------------------------------------------------------
// 2. Structural equality — same values
// ---------------------------------------------------------------------------

#[test]
fn test_record_equality_same_values() {
    let src = r#"
record Color {
    r: i64,
    g: i64,
    b: i64
}
fn check() -> i64 {
    let a = Color { r: 255, g: 128, b: 0 };
    let b = Color { r: 255, g: 128, b: 0 };
    if a == b {
        return 1;
    }
    0
}
fn main() -> i64 {
    check()
}
"#;
    assert_eq!(eval_str(src), "1");
    assert_eq!(mir_str(src), "1");
}

// ---------------------------------------------------------------------------
// 3. Structural equality — different values
// ---------------------------------------------------------------------------

#[test]
fn test_record_equality_different_values() {
    let src = r#"
record Vec2 {
    x: f64,
    y: f64
}
fn check() -> i64 {
    let a = Vec2 { x: 1.0, y: 2.0 };
    let b = Vec2 { x: 3.0, y: 4.0 };
    if a != b {
        return 1;
    }
    0
}
fn main() -> i64 {
    check()
}
"#;
    assert_eq!(eval_str(src), "1");
    assert_eq!(mir_str(src), "1");
}

// ---------------------------------------------------------------------------
// 4. Compile-time immutability — type checker emits E0160
// ---------------------------------------------------------------------------

#[test]
fn test_record_immutability_rejected_e0160() {
    let src = r#"
record Config {
    width: i64,
    height: i64
}
fn main() -> i64 {
    let mut c = Config { width: 800, height: 600 };
    c.width = 1024;
    c.width
}
"#;
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors(), "parse errors");

    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    let has_e0160 = checker.diagnostics.diagnostics.iter()
        .any(|d| d.code == "E0160");
    assert!(
        has_e0160,
        "expected E0160 for record field assignment, got: {:?}",
        checker.diagnostics.diagnostics
    );
}

// ---------------------------------------------------------------------------
// 5. Runtime immutability — eval
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

// ---------------------------------------------------------------------------
// 6. Runtime immutability — MIR-exec
// ---------------------------------------------------------------------------

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
// 7. Nested records
// ---------------------------------------------------------------------------

#[test]
fn test_record_nested() {
    let src = r#"
record Point {
    x: f64,
    y: f64
}
record Line {
    start: Any,
    end: Any
}
fn main() -> f64 {
    let p1 = Point { x: 1.0, y: 2.0 };
    let p2 = Point { x: 4.0, y: 6.0 };
    let line = Line { start: p1, end: p2 };
    line.end.x - line.start.x
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity: eval={ev} mir={mv}");
    assert_eq!(ev, "3");
}

// ---------------------------------------------------------------------------
// 8. Record as function argument
// ---------------------------------------------------------------------------

#[test]
fn test_record_as_fn_arg() {
    let src = r#"
record Rect {
    w: i64,
    h: i64
}
fn area(r: Any) -> i64 {
    r.w * r.h
}
fn main() -> i64 {
    let r = Rect { w: 10, h: 20 };
    area(r)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "200");
}

// ---------------------------------------------------------------------------
// 9. Parity — comprehensive eval vs MIR
// ---------------------------------------------------------------------------

#[test]
fn test_record_parity_comprehensive() {
    let src = r#"
record Data {
    a: i64,
    b: f64
}
fn process(d: Any) -> f64 {
    float(d.a) + d.b
}
fn main() -> f64 {
    let d = Data { a: 10, b: 0.5 };
    process(d)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity: eval={ev} mir={mv}");
    assert_eq!(ev, "10.5");
}

// ---------------------------------------------------------------------------
// 10. Struct fields still mutable (contrast with records)
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
// 11. Record vs struct semantics — same fields, different behavior
// ---------------------------------------------------------------------------

#[test]
fn test_record_vs_struct_semantics() {
    // Struct: mutable, so field assignment succeeds
    let struct_src = r#"
struct Pair {
    x: i64,
    y: i64
}
fn main() -> i64 {
    let mut p = Pair { x: 1, y: 2 };
    p.x = 42;
    p.x
}
"#;
    assert_eq!(eval_str(struct_src), "42");

    // Record: immutable, so field assignment fails at runtime
    let record_src = r#"
record Pair {
    x: i64,
    y: i64
}
fn main() -> i64 {
    let mut p = Pair { x: 1, y: 2 };
    p.x = 42;
    p.x
}
"#;
    let program = parse(record_src);
    let mut interp = cjc_eval::Interpreter::new(42);
    assert!(interp.exec(&program).is_err(), "record mutation should fail");
}

// ---------------------------------------------------------------------------
// 12. Record match destructuring
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
// 13. Determinism — same result across runs
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
    assert_eq!(v1, v2, "eval deterministic");
    assert_eq!(m1, m2, "MIR deterministic");
    assert_eq!(v1, m1, "cross-executor deterministic");
}

// ---------------------------------------------------------------------------
// 14. Different record types are not equal
// ---------------------------------------------------------------------------

#[test]
fn test_different_record_types_not_equal() {
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
    assert_eq!(eval_str(src), "0", "different record types should not be equal");
}

// ---------------------------------------------------------------------------
// 15. Record with multiple fields — ordering preserved
// ---------------------------------------------------------------------------

#[test]
fn test_record_multi_field_access() {
    let src = r#"
record Entry {
    name: str,
    age: i64,
    score: f64
}
fn main() -> f64 {
    let e = Entry { name: "Alice", age: 30, score: 95.5 };
    float(e.age) + e.score
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "125.5");
}
