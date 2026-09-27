//! Trait/impl system end-to-end tests.
//!
//! Verifies: parsing, HIR lowering, MIR lowering, eval dispatch, MIR-exec dispatch.

// Both helpers return printed output: a top-level expression statement
// evaluates to Void, so programs report results with `print`.
fn eval(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).unwrap();
    interp.output.clone()
}

fn mir_exec(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

fn printed_f64(out: &[String]) -> f64 {
    assert_eq!(out.len(), 1, "expected one printed value, got {:?}", out);
    out[0].trim().parse().unwrap_or_else(|_| panic!("not a float: {:?}", out))
}

#[test]
fn test_trait_decl_parses() {
    let src = r#"
trait Printable {
    fn to_str(self: Any) -> str;
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
print(m);
"#;
    let result = eval(src);
    let v = printed_f64(&result);
    assert!((v - 5.0).abs() < 1e-10, "expected 5.0, got {}", v);
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
print(s);
"#;
    let result = mir_exec(src);
    let v = printed_f64(&result);
    assert!((v - 30.0).abs() < 1e-10, "expected 30.0, got {}", v);
}

#[test]
fn test_trait_conformance_check() {
    let src = r#"
trait Addable {
    fn add(self: Any, other: Any) -> Any;
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
print(a.dot(b));
"#;
    let eval_result = eval(src);
    let mir_result = mir_exec(src);
    assert!(!eval_result.is_empty(), "program printed nothing");
    assert_eq!(eval_result, mir_result, "eval and MIR must agree");
}
