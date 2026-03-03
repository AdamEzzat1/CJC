//! Role 9, Requirement 2: Traits Over Inheritance
//!
//! CJC has NO inheritance (extends/super absent by design).  All polymorphism
//! comes through traits + impl blocks.  Tests verify:
//!   - Trait declaration parsing
//!   - `impl Trait for Type` syntax
//!   - Bare `impl Type` methods
//!   - Trait bound enforcement (`<T: Trait>`)
//!   - No `extends` or `super` keywords
//!   - Method dispatch (qualified and dot)
//!   - Multiple trait implementations
//!   - Parity between eval and MIR-exec

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
// 1. Trait declaration parses
// ---------------------------------------------------------------------------

#[test]
fn test_trait_decl_parses() {
    let src = r#"
trait Drawable {
    fn draw(self: Any) -> i64;
}
fn main() -> i64 { 0 }
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "trait decl should parse: {:?}", diags.diagnostics);

    // Verify the AST contains a TraitDecl
    let has_trait = program.declarations.iter().any(|d| {
        matches!(&d.kind, cjc_ast::DeclKind::Trait(_))
    });
    assert!(has_trait, "expected a TraitDecl in the AST");
}

// ---------------------------------------------------------------------------
// 2. impl Trait for Type syntax
// ---------------------------------------------------------------------------

#[test]
fn test_impl_trait_for_type() {
    let src = r#"
trait Describe {
    fn desc(self: Any) -> str;
}
struct Dog {
    name: str
}
impl Describe for Dog {
    fn desc(self: Dog) -> str {
        self.name
    }
}
fn main() -> str {
    let d = Dog { name: "Rex" };
    Dog.desc(d)
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "impl Trait for Type should parse: {:?}", diags.diagnostics);

    // Verify the AST contains an ImplDecl
    let has_impl = program.declarations.iter().any(|d| {
        matches!(&d.kind, cjc_ast::DeclKind::Impl(_))
    });
    assert!(has_impl, "expected ImplDecl in the AST");
}

// ---------------------------------------------------------------------------
// 3. Bare impl methods (no trait)
// ---------------------------------------------------------------------------

#[test]
fn test_impl_bare_methods() {
    let src = r#"
struct Counter {
    value: i64
}
impl Counter {
    fn get(self: Counter) -> i64 {
        self.value
    }
    fn inc(self: Counter) -> i64 {
        self.value + 1
    }
}
fn main() -> i64 {
    let c = Counter { value: 10 };
    Counter.get(c) + Counter.inc(c)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "21");
}

// ---------------------------------------------------------------------------
// 4. Trait bound enforcement — satisfied
// ---------------------------------------------------------------------------

#[test]
fn test_trait_bound_satisfied() {
    let src = r#"
trait Numeric {}
impl Numeric for i64 {}
fn add_nums<T: Numeric>(a: T, b: T) -> T { a }
let x = add_nums(1, 2);
"#;
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors(), "parse errors: {:?}", parse_diags.diagnostics);

    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    let bound_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code.starts_with("E03") || d.code.starts_with("E6"))
        .collect();
    assert!(bound_errors.is_empty(), "unexpected bound errors: {:?}", bound_errors);
}

// ---------------------------------------------------------------------------
// 5. Trait bound enforcement — violated
// ---------------------------------------------------------------------------

#[test]
fn test_trait_bound_violated() {
    let src = r#"
trait Numeric {}
impl Numeric for i64 {}
fn add_nums<T: Numeric>(a: T, b: T) -> T { a }
let x = add_nums("hello", "world");
"#;
    let (program, parse_diags) = cjc_parser::parse_source(src);
    assert!(!parse_diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    let bound_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.message.contains("trait bound") || d.code == "E0300")
        .collect();
    assert!(!bound_errors.is_empty(),
        "expected bound violation, got: {:?}", checker.diagnostics.diagnostics);
}

// ---------------------------------------------------------------------------
// 6. No `extends` keyword — parse error
// ---------------------------------------------------------------------------

#[test]
fn test_no_extends_keyword() {
    let src = r#"
struct A {
    x: i64
}
struct B extends A {
    y: i64
}
fn main() -> i64 { 0 }
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(
        diags.has_errors(),
        "CJC should not support `extends` keyword"
    );
}

// ---------------------------------------------------------------------------
// 7. Method dispatch — qualified (Type.method)
// ---------------------------------------------------------------------------

#[test]
fn test_method_dispatch_qualified() {
    let src = r#"
struct Point {
    x: f64,
    y: f64
}
impl Point {
    fn magnitude_sq(self: Point) -> f64 {
        self.x * self.x + self.y * self.y
    }
}
fn main() -> f64 {
    let p = Point { x: 3.0, y: 4.0 };
    Point.magnitude_sq(p)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "25");
}

// ---------------------------------------------------------------------------
// 8. Method dispatch — dot syntax (p.method())
// ---------------------------------------------------------------------------

#[test]
fn test_method_dispatch_dot() {
    let src = r#"
struct Acc {
    total: i64
}
impl Acc {
    fn get_total(self: Acc) -> i64 {
        self.total
    }
}
fn main() -> i64 {
    let a = Acc { total: 42 };
    a.get_total()
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "42");
}

// ---------------------------------------------------------------------------
// 9. Multiple trait implementations for same type
// ---------------------------------------------------------------------------

#[test]
fn test_impl_multiple_traits() {
    let src = r#"
trait Alpha {
    fn a_val(self: Any) -> i64;
}
trait Beta {
    fn b_val(self: Any) -> i64;
}
struct Both {
    x: i64
}
impl Alpha for Both {
    fn a_val(self: Both) -> i64 {
        self.x + 1
    }
}
impl Beta for Both {
    fn b_val(self: Both) -> i64 {
        self.x + 2
    }
}
fn main() -> i64 {
    let b = Both { x: 10 };
    Both.a_val(b) + Both.b_val(b)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "23");
}

// ---------------------------------------------------------------------------
// 10. Trait bound error message quality
// ---------------------------------------------------------------------------

#[test]
fn test_trait_bound_error_message() {
    let src = r#"
trait Numeric {}
impl Numeric for i64 {}
fn square<T: Numeric>(x: T) -> T { x }
let s = square("oops");
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    let bound_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.message.contains("trait bound") || d.code == "E0300")
        .collect();
    assert!(!bound_errors.is_empty(), "expected trait bound error");
}

// ---------------------------------------------------------------------------
// 11. Parity — trait dispatch
// ---------------------------------------------------------------------------

#[test]
fn test_parity_trait_dispatch() {
    let src = r#"
struct Calculator {
    base: i64
}
impl Calculator {
    fn add(self: Calculator, n: i64) -> i64 {
        self.base + n
    }
}
fn main() -> i64 {
    let c = Calculator { base: 100 };
    Calculator.add(c, 42)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity: eval={ev} mir={mv}");
    assert_eq!(ev, "142");
}

// ---------------------------------------------------------------------------
// 12. impl Type : Trait syntax (alternative CJC syntax)
// ---------------------------------------------------------------------------

#[test]
fn test_impl_colon_trait_syntax() {
    let src = r#"
trait Greetable {
    fn greet(self: Any) -> i64;
}
struct Person {
    age: i64
}
impl Person : Greetable {
    fn greet(self: Person) -> i64 {
        self.age
    }
}
fn main() -> i64 {
    let p = Person { age: 25 };
    Person.greet(p)
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "impl Type : Trait syntax should parse: {:?}", diags.diagnostics);

    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "25");
}
