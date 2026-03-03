//! v0.1 Contract Tests: Type System Reality
//!
//! Locks down: generics parse, bounds parse, Any polymorphism,
//! type annotations required, deterministic dispatch, eval/MIR parity.

// ── Helpers ──────────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn eval_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).unwrap();
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

// ── Tests ────────────────────────────────────────────────────────

#[test]
fn generics_parse() {
    // struct Pair<T, U> should parse without errors
    let src = r#"
struct Pair<T, U> {
    first: T,
    second: U,
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "generic struct should parse: {:?}", diags.diagnostics);
}

#[test]
fn generic_with_bounds_parse() {
    // struct Foo<T: Clone> should parse
    let src = r#"
struct Bounded<T: Clone> {
    value: T,
}
"#;
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "bounded generic should parse: {:?}", diags.diagnostics);
}

#[test]
fn any_polymorphism() {
    // fn f(x: Any) accepts i64, f64, str at runtime
    let src = r#"
fn show(x: Any) -> str {
    to_string(x)
}
print(show(42));
print(show(3.14));
print(show("hello"));
"#;
    let out = eval_output(src);
    assert_eq!(out.len(), 3, "should produce 3 outputs");
    assert_eq!(out[0], "42");
    assert_eq!(out[1], "3.14");
    assert_eq!(out[2], "hello");
}

#[test]
fn type_annotations_required() {
    // fn f(x) without annotation should parse error
    let bad = "fn f(x) -> i64 { x }";
    let (_, diags) = cjc_parser::parse_source(bad);
    assert!(diags.has_errors(), "fn param without type annotation should fail to parse");
}

#[test]
fn dynamic_dispatch_deterministic() {
    // Same inputs → same dispatch results, always
    let src = r#"
fn identity(x: Any) -> Any { x }
let a = identity(1);
let b = identity(2);
let c = identity(3);
print(a);
print(b);
print(c);
"#;
    let out1 = eval_output(src);
    let out2 = eval_output(src);
    assert_eq!(out1, out2, "dispatch must be deterministic across runs");
    assert_eq!(out1, vec!["1", "2", "3"]);
}

#[test]
fn parity_struct_any_fields() {
    // eval and MIR should produce identical results for struct with Any fields
    let src = r#"
struct Box {
    value: Any,
}
let b = Box { value: 42 };
print(b.value);
"#;
    let eval_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(eval_out, mir_out, "struct Any parity: eval vs MIR must match");
}
