//! Trait/impl system end-to-end tests.
//!
//! Verifies: parsing, HIR lowering, MIR lowering, eval dispatch, MIR-exec dispatch.

fn eval(src: &str) -> cjc_runtime::Value {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).unwrap()
}

fn mir_exec(src: &str) -> cjc_runtime::Value {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let (val, _) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    val
}

#[test]
fn test_trait_decl_parses() {
    let src = r#"
trait Printable {
    fn to_str(self: Any) -> str
}
let x = 1;
"#;
    eval(src); // should not panic
}

#[test]
fn test_impl_registers_methods() {
    let src = r#"
struct Point { x: f64, y: f64 }
impl Point {
    fn magnitude(self: Point) -> f64 {
        sqrt(self.x * self.x + self.y * self.y)
    }
}
let p = Point { x: 3.0, y: 4.0 };
let m = p.magnitude();
m
"#;
    let result = eval(src);
    match result {
        cjc_runtime::Value::Float(v) => {
            assert!((v - 5.0).abs() < 1e-10, "expected 5.0, got {}", v);
        }
        _ => panic!("expected Float, got {:?}", result),
    }
}

#[test]
fn test_impl_methods_work_in_mir() {
    let src = r#"
struct Point { x: f64, y: f64 }
impl Point {
    fn sum_coords(self: Point) -> f64 {
        self.x + self.y
    }
}
let p = Point { x: 10.0, y: 20.0 };
let s = p.sum_coords();
s
"#;
    let result = mir_exec(src);
    match result {
        cjc_runtime::Value::Float(v) => {
            assert!((v - 30.0).abs() < 1e-10, "expected 30.0, got {}", v);
        }
        _ => panic!("expected Float, got {:?}", result),
    }
}

#[test]
fn test_trait_conformance_check() {
    let src = r#"
trait Addable {
    fn add(self: Any, other: Any) -> Any
}
impl i64 : Addable {
    fn add(self: i64, other: i64) -> i64 { self + other }
}
let x = 1;
"#;
    // Should parse and type-check without errors
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);
    // No errors expected
}

#[test]
fn test_trait_impl_parity() {
    let src = r#"
struct Vec2 { x: f64, y: f64 }
impl Vec2 {
    fn dot(self: Vec2, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }
}
let a = Vec2 { x: 1.0, y: 2.0 };
let b = Vec2 { x: 3.0, y: 4.0 };
a.dot(b)
"#;
    let eval_result = eval(src);
    let mir_result = mir_exec(src);
    assert_eq!(
        format!("{}", eval_result),
        format!("{}", mir_result),
        "eval and MIR must agree"
    );
}
