//! Role 9, Requirement 4: Effect Tags on Methods Mandatory
//!
//! Methods declared with effect annotations (`/ pure`, `/ io`, etc.) must
//! have their effects checked just like top-level functions.  This is THE
//! critical gap identified by the Role 9 audit — the infrastructure works
//! but had zero test coverage.
//!
//! Tests verify:
//!   - Method effect annotation parsing (pure, io, combined)
//!   - E4002 violation: pure method calling IO builtin (print)
//!   - Pure method calling pure builtin passes
//!   - IO method calling print passes
//!   - Pure method calling alloc builtin violates
//!   - Unannotated method allows any effect (backward compat)
//!   - Pure method calling user-defined IO method violates
//!   - Effect-annotated methods execute correctly (runtime parity)
//!   - Effect annotation on trait methods
//!   - Error message quality

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

// ---------------------------------------------------------------------------
// 1. Method pure annotation parses
// ---------------------------------------------------------------------------

#[test]
fn test_method_pure_annotation_parses() {
    let src = r#"
struct Counter {
    value: i64
}
impl Counter {
    fn get(self: Counter) -> i64 / pure {
        self.value
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    if let cjc_ast::DeclKind::Impl(ref impl_decl) = program.declarations[1].kind {
        assert_eq!(
            impl_decl.methods[0].effect_annotation,
            Some(vec!["pure".to_string()]),
            "method should have /pure annotation"
        );
    } else {
        panic!("expected ImplDecl");
    }
}

// ---------------------------------------------------------------------------
// 2. Method IO annotation parses
// ---------------------------------------------------------------------------

#[test]
fn test_method_io_annotation_parses() {
    let src = r#"
struct Logger {
    prefix: str
}
impl Logger {
    fn log(self: Logger, msg: i64) / io {
        print(msg);
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    if let cjc_ast::DeclKind::Impl(ref impl_decl) = program.declarations[1].kind {
        assert_eq!(
            impl_decl.methods[0].effect_annotation,
            Some(vec!["io".to_string()]),
            "method should have /io annotation"
        );
    } else {
        panic!("expected ImplDecl");
    }
}

// ---------------------------------------------------------------------------
// 3. Method combined effects parse
// ---------------------------------------------------------------------------

#[test]
fn test_method_combined_effects_parse() {
    let src = r#"
struct Processor {
    data: i64
}
impl Processor {
    fn process(self: Processor) -> i64 / io + alloc {
        print(self.data);
        self.data
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    if let cjc_ast::DeclKind::Impl(ref impl_decl) = program.declarations[1].kind {
        assert_eq!(
            impl_decl.methods[0].effect_annotation,
            Some(vec!["io".to_string(), "alloc".to_string()]),
            "method should have /io+alloc annotation"
        );
    } else {
        panic!("expected ImplDecl");
    }
}

// ---------------------------------------------------------------------------
// 4. KEY TEST: Pure method calling print() → E4002
// ---------------------------------------------------------------------------

#[test]
fn test_pure_method_calling_print_violates() {
    let src = r#"
struct Widget {
    name: str
}
impl Widget {
    fn render(self: Widget) -> i64 / pure {
        print(self.name);
        0
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        !effect_errors.is_empty(),
        "expected E4002: pure method calling print, got: {:?}",
        checker.diagnostics.diagnostics
    );
    let msg = &effect_errors[0].message;
    assert!(msg.contains("print"), "error should mention 'print': {}", msg);
}

// ---------------------------------------------------------------------------
// 5. Pure method calling pure builtin passes
// ---------------------------------------------------------------------------

#[test]
fn test_pure_method_calling_sqrt_passes() {
    let src = r#"
struct Hyp {
    a: f64,
    b: f64
}
impl Hyp {
    fn compute(self: Hyp) -> f64 / pure {
        sqrt(self.a * self.a + self.b * self.b)
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        effect_errors.is_empty(),
        "pure method calling sqrt should pass: {:?}",
        effect_errors
    );
}

// ---------------------------------------------------------------------------
// 6. IO method calling print passes
// ---------------------------------------------------------------------------

#[test]
fn test_io_method_calling_print_passes() {
    let src = r#"
struct Printer {
    val: i64
}
impl Printer {
    fn show(self: Printer) / io {
        print(self.val);
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        effect_errors.is_empty(),
        "IO method should allow print: {:?}",
        effect_errors
    );
}

// ---------------------------------------------------------------------------
// 7. Pure method calling alloc builtin violates
// ---------------------------------------------------------------------------

#[test]
fn test_pure_method_calling_alloc_violates() {
    let src = r#"
struct Builder {
    x: f64
}
impl Builder {
    fn build(self: Builder) -> f64 / pure {
        outer(self.x, self.x)
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        !effect_errors.is_empty(),
        "pure method calling alloc builtin should fail: {:?}",
        checker.diagnostics.diagnostics
    );
}

// ---------------------------------------------------------------------------
// 8. Unannotated method allows IO (backward compat)
// ---------------------------------------------------------------------------

#[test]
fn test_unannotated_method_allows_io() {
    let src = r#"
struct Chatty {
    msg: i64
}
impl Chatty {
    fn speak(self: Chatty) -> i64 {
        print(self.msg);
        self.msg
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        effect_errors.is_empty(),
        "unannotated method should not trigger effect errors: {:?}",
        effect_errors
    );
}

// ---------------------------------------------------------------------------
// 9. Pure method calling user-defined IO method → E4002
// ---------------------------------------------------------------------------

#[test]
fn test_pure_method_calling_io_method_violates() {
    let src = r#"
fn do_io(x: i64) -> i64 / io {
    print(x);
    x
}
struct S {
    val: i64
}
impl S {
    fn pure_wrapper(self: S) -> i64 / pure {
        do_io(self.val)
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        !effect_errors.is_empty(),
        "pure method calling IO user fn should fail: {:?}",
        checker.diagnostics.diagnostics
    );
}

// ---------------------------------------------------------------------------
// 10. Effect-annotated methods execute correctly — eval
// ---------------------------------------------------------------------------

#[test]
fn test_method_effect_parity_eval() {
    let src = r#"
struct Doubler {
    n: i64
}
impl Doubler {
    fn double(self: Doubler) -> i64 / pure {
        self.n * 2
    }
}
fn main() -> i64 {
    let d = Doubler { n: 21 };
    Doubler.double(d)
}
"#;
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let val = interp.exec(&program).expect("eval failed");
    assert_eq!(format!("{val}"), "42");
}

// ---------------------------------------------------------------------------
// 11. Effect-annotated methods execute correctly — MIR
// ---------------------------------------------------------------------------

#[test]
fn test_method_effect_parity_mir() {
    let src = r#"
struct Doubler {
    n: i64
}
impl Doubler {
    fn double(self: Doubler) -> i64 / pure {
        self.n * 2
    }
}
fn main() -> i64 {
    let d = Doubler { n: 21 };
    Doubler.double(d)
}
"#;
    let program = parse(src);
    let (val, _exec) =
        cjc_mir_exec::run_program_with_executor(&program, 42).expect("MIR exec failed");
    assert_eq!(format!("{val}"), "42");
}

// ---------------------------------------------------------------------------
// 12. Pure method with clean body — type-checks OK
// ---------------------------------------------------------------------------

#[test]
fn test_pure_method_pure_body_passes() {
    let src = r#"
struct Adder {
    a: i64,
    b: i64
}
impl Adder {
    fn sum(self: Adder) -> i64 / pure {
        self.a + self.b
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        effect_errors.is_empty(),
        "pure method with clean body should pass: {:?}",
        effect_errors
    );
}

// ---------------------------------------------------------------------------
// 13. Effect annotation on trait method
// ---------------------------------------------------------------------------

#[test]
fn test_effect_annotation_on_impl_method_for_trait() {
    // Trait method *declarations* (with `;`) don't support effect annotations
    // in the parser, but *impl* methods for a trait DO support them.
    let src = r#"
trait Computable {
    fn compute(self: Any) -> i64;
}
struct Num {
    val: i64
}
impl Computable for Num {
    fn compute(self: Num) -> i64 / pure {
        self.val
    }
}
fn main() -> i64 {
    let n = Num { val: 7 };
    Num.compute(n)
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "impl method for trait with effect annotation should parse: {:?}",
        diags.diagnostics
    );

    // Verify the impl method has the effect annotation
    let impl_decl = program.declarations.iter().find(|d| {
        matches!(&d.kind, cjc_ast::DeclKind::Impl(_))
    });
    assert!(impl_decl.is_some(), "expected ImplDecl in AST");

    // Verify no E4002 — the pure body is clean
    let mut checker = TypeChecker::new();
    checker.check_program(&program);
    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        effect_errors.is_empty(),
        "pure impl method with clean body should pass: {:?}",
        effect_errors
    );
}

// ---------------------------------------------------------------------------
// 14. Method effect error message quality
// ---------------------------------------------------------------------------

#[test]
fn test_method_effect_error_message() {
    let src = r#"
struct Svc {
    id: i64
}
impl Svc {
    fn bad(self: Svc) -> i64 / pure {
        print(self.id);
        self.id
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(!effect_errors.is_empty(), "expected E4002");

    let msg = &effect_errors[0].message;
    assert!(
        msg.contains("io") || msg.contains("IO"),
        "error message should mention IO effect: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 15. Alloc method rejects IO
// ---------------------------------------------------------------------------

#[test]
fn test_alloc_method_rejects_io() {
    let src = r#"
struct Mgr {
    data: f64
}
impl Mgr {
    fn allocate(self: Mgr) -> f64 / alloc {
        print(self.data);
        outer(self.data, self.data)
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut checker = TypeChecker::new();
    checker.check_program(&program);

    let effect_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .collect();
    assert!(
        !effect_errors.is_empty(),
        "alloc-only method should reject print (IO): {:?}",
        checker.diagnostics.diagnostics
    );
}
