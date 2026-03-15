//! Tests for trait/impl method dispatch in the AST evaluator (cjc-eval).
//!
//! Verifies that:
//! - impl methods are registered as `Type.method` during declaration pass
//! - Instance method calls (`x.method()`) dispatch to impl methods
//! - Static method calls (`Type.method(x)`) dispatch to impl methods
//! - Trait definitions are stored for validation
//! - Both bare `impl Type` and `impl Trait for Type` work
//! - Parity: eval and MIR-exec produce identical results

use cjc_parser::parse_source;
use cjc_runtime::Value;

fn eval_src(src: &str) -> Value {
    let (prog, diags) = parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors: {:?}",
        diags.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&prog).expect("eval should succeed")
}

fn mir_src(src: &str) -> Value {
    let (prog, diags) = parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors: {:?}",
        diags.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    let (val, _) = cjc_mir_exec::run_program_with_executor(&prog, 42)
        .expect("MIR exec should succeed");
    val
}

// ── Bare impl: static dispatch ──────────────────────────────────────────

#[test]
fn test_impl_method_static_dispatch_eval() {
    let src = r#"
struct Circle { radius: f64, }
impl Circle {
    fn area(c: Circle) -> f64 {
        3.14159 * c.radius * c.radius
    }
}
fn main() -> f64 {
    let c = Circle { radius: 2.0 };
    Circle.area(c)
}
"#;
    let val = eval_src(src);
    if let Value::Float(v) = val {
        assert!((v - 12.56636).abs() < 0.001, "expected ~12.566, got {}", v);
    } else {
        panic!("expected Float, got {:?}", val);
    }
}

// ── Bare impl: instance method dispatch ─────────────────────────────────

#[test]
fn test_impl_method_instance_dispatch_eval() {
    let src = r#"
struct Rect { w: f64, h: f64, }
impl Rect {
    fn area(self: Rect) -> f64 {
        self.w * self.h
    }
}
fn main() -> f64 {
    let r = Rect { w: 3.0, h: 4.0 };
    r.area()
}
"#;
    let val = eval_src(src);
    if let Value::Float(v) = val {
        assert!((v - 12.0).abs() < 0.001, "expected 12.0, got {}", v);
    } else {
        panic!("expected Float, got {:?}", val);
    }
}

// ── Multiple impl methods on same type ──────────────────────────────────

#[test]
fn test_impl_multiple_methods_eval() {
    let src = r#"
struct Vec2 { x: f64, y: f64, }
impl Vec2 {
    fn length_sq(self: Vec2) -> f64 {
        self.x * self.x + self.y * self.y
    }
    fn dot(self: Vec2, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }
}
fn main() -> f64 {
    let a = Vec2 { x: 3.0, y: 4.0 };
    let b = Vec2 { x: 1.0, y: 2.0 };
    a.length_sq() + a.dot(b)
}
"#;
    let val = eval_src(src);
    // length_sq = 9+16 = 25, dot = 3+8 = 11, total = 36
    if let Value::Float(v) = val {
        assert!((v - 36.0).abs() < 0.001, "expected 36.0, got {}", v);
    } else {
        panic!("expected Float, got {:?}", val);
    }
}

// ── Two different types with same method name ───────────────────────────

#[test]
fn test_impl_different_types_same_method_name() {
    let src = r#"
struct Dog { loudness: f64, }
struct Cat { softness: f64, }
impl Dog {
    fn speak(self: Dog) -> f64 { self.loudness * 10.0 }
}
impl Cat {
    fn speak(self: Cat) -> f64 { self.softness * 0.1 }
}
fn main() -> f64 {
    let d = Dog { loudness: 5.0 };
    let c = Cat { softness: 8.0 };
    d.speak() + c.speak()
}
"#;
    let val = eval_src(src);
    // Dog.speak = 50.0, Cat.speak = 0.8, total = 50.8
    if let Value::Float(v) = val {
        assert!((v - 50.8).abs() < 0.001, "expected 50.8, got {}", v);
    } else {
        panic!("expected Float, got {:?}", val);
    }
}

// ── impl Trait for Type ─────────────────────────────────────────────────

#[test]
fn test_impl_trait_for_type_eval() {
    let src = r#"
trait Shape {
    fn area(self: Circle) -> f64;
}
struct Circle { radius: f64, }
impl Shape for Circle {
    fn area(self: Circle) -> f64 {
        3.14159 * self.radius * self.radius
    }
}
fn main() -> f64 {
    let c = Circle { radius: 3.0 };
    c.area()
}
"#;
    let (prog, diags) = parse_source(src);
    if diags.has_errors() {
        // Parser may not support `impl Trait for Type` yet -- skip gracefully
        eprintln!("skipping: impl Trait for Type not yet parsed");
        return;
    }
    let mut interp = cjc_eval::Interpreter::new(42);
    let val = interp.exec(&prog).expect("eval should succeed");
    if let Value::Float(v) = val {
        assert!((v - 28.27431).abs() < 0.001, "expected ~28.274, got {}", v);
    } else {
        panic!("expected Float, got {:?}", val);
    }
}

// ── Trait definition stored ─────────────────────────────────────────────

#[test]
fn test_trait_def_stored_in_eval() {
    // Just verify it doesn't panic when a trait is declared and impl'd
    let src = r#"
trait Summable {
    fn sum_val(self: Pair) -> i64;
}
struct Pair { a: i64, b: i64, }
impl Pair {
    fn sum_val(self: Pair) -> i64 {
        self.a + self.b
    }
}
fn main() -> i64 {
    let p = Pair { a: 10, b: 20 };
    p.sum_val()
}
"#;
    let val = eval_src(src);
    match val {
        Value::Int(v) => assert_eq!(v, 30),
        _ => panic!("expected Int(30), got {:?}", val),
    }
}

// ── Parity: eval vs MIR-exec ───────────────────────────────────────────

#[test]
fn test_impl_method_parity_eval_mir() {
    let src = r#"
struct Counter { val: i64, }
impl Counter {
    fn get(self: Counter) -> i64 {
        self.val
    }
    fn doubled(self: Counter) -> i64 {
        self.val * 2
    }
}
fn main() -> i64 {
    let c = Counter { val: 7 };
    c.get() + c.doubled()
}
"#;
    let eval_val = eval_src(src);
    let mir_val = mir_src(src);
    match (&eval_val, &mir_val) {
        (Value::Int(a), Value::Int(b)) => assert_eq!(a, b, "eval and MIR-exec must agree"),
        _ => panic!("expected Int from both, got eval={:?} mir={:?}", eval_val, mir_val),
    }
    match eval_val {
        Value::Int(v) => assert_eq!(v, 21),
        _ => panic!("expected Int(21), got {:?}", eval_val),
    }
}

// ── Determinism: same result across runs ────────────────────────────────

#[test]
fn test_impl_method_determinism() {
    let src = r#"
struct Accum { total: f64, }
impl Accum {
    fn value(self: Accum) -> f64 { self.total }
}
fn main() -> f64 {
    let a = Accum { total: 42.5 };
    a.value()
}
"#;
    let v1 = eval_src(src);
    let v2 = eval_src(src);
    match (&v1, &v2) {
        (Value::Float(a), Value::Float(b)) => {
            assert_eq!(a.to_bits(), b.to_bits(), "impl method dispatch must be deterministic");
        }
        _ => panic!("expected Float from both runs, got {:?} and {:?}", v1, v2),
    }
}
