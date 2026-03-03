//! LH04: Enforced Effect Typing tests
//!
//! Verifies:
//! - Effect annotation parsing (`fn foo() -> i64 / pure { ... }`)
//! - Effect checking detects violations (IO in pure context, etc.)
//! - Multiple effects can be combined (`/ io + alloc`)
//! - Unannotated functions remain backward-compatible (any effect)
//! - `nogc fn` interop with effect annotations
//! - Effect registry integration

use cjc_types::{TypeChecker, EffectSet};

// ── Parsing: effect annotations ─────────────────────────────────

#[test]
fn test_parse_pure_annotation() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 / pure {
    a + b
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    if let cjc_ast::DeclKind::Fn(ref f) = program.declarations[0].kind {
        assert_eq!(f.effect_annotation, Some(vec!["pure".to_string()]));
    } else {
        panic!("expected FnDecl");
    }
}

#[test]
fn test_parse_io_annotation() {
    let src = r#"
fn greet(name: str) / io {
    print(name);
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    if let cjc_ast::DeclKind::Fn(ref f) = program.declarations[0].kind {
        assert_eq!(f.effect_annotation, Some(vec!["io".to_string()]));
    } else {
        panic!("expected FnDecl");
    }
}

#[test]
fn test_parse_multiple_effects() {
    let src = r#"
fn process(data: i64) -> i64 / io + alloc {
    print(data);
    data
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    if let cjc_ast::DeclKind::Fn(ref f) = program.declarations[0].kind {
        assert_eq!(
            f.effect_annotation,
            Some(vec!["io".to_string(), "alloc".to_string()])
        );
    } else {
        panic!("expected FnDecl");
    }
}

#[test]
fn test_parse_no_annotation() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 {
    a + b
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    if let cjc_ast::DeclKind::Fn(ref f) = program.declarations[0].kind {
        assert_eq!(f.effect_annotation, None);
    } else {
        panic!("expected FnDecl");
    }
}

// ── Effect checking: violations ─────────────────────────────────

#[test]
fn test_pure_fn_with_io_call_violates() {
    let src = r#"
fn pure_add(a: i64, b: i64) -> i64 / pure {
    print(a);
    a + b
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
        "expected E4002 effect violation, got: {:?}",
        checker.diagnostics.diagnostics
    );
    // Verify the error message mentions "print" and "io"
    let msg = &effect_errors[0].message;
    assert!(msg.contains("print"), "error should mention 'print': {}", msg);
    assert!(msg.contains("io"), "error should mention 'io': {}", msg);
}

#[test]
fn test_pure_fn_without_io_passes() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 / pure {
    a + b
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
        "unexpected effect errors: {:?}",
        effect_errors
    );
}

#[test]
fn test_io_fn_with_print_passes() {
    let src = r#"
fn greet(x: i64) / io {
    print(x);
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
        "IO fn should allow print: {:?}",
        effect_errors
    );
}

#[test]
fn test_pure_fn_calling_pure_builtin_passes() {
    let src = r#"
fn compute(x: f64) -> f64 / pure {
    sqrt(x)
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
        "pure fn calling sqrt should pass: {:?}",
        effect_errors
    );
}

#[test]
fn test_pure_fn_calling_alloc_builtin_violates() {
    let src = r#"
fn make_stuff(x: f64) -> f64 / pure {
    outer(x, x)
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
        "pure fn calling alloc builtin should fail: {:?}",
        checker.diagnostics.diagnostics
    );
}

// ── Backward compatibility ──────────────────────────────────────

#[test]
fn test_unannotated_fn_allows_anything() {
    let src = r#"
fn messy(x: i64) -> i64 {
    print(x);
    x
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
        "unannotated fn should not trigger effect errors: {:?}",
        effect_errors
    );
}

// ── Combined effects ────────────────────────────────────────────

#[test]
fn test_io_alloc_fn_allows_both() {
    let src = r#"
fn do_stuff(x: f64) -> f64 / io + alloc {
    print(x);
    outer(x, x)
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
        "io+alloc fn should allow print and outer: {:?}",
        effect_errors
    );
}

#[test]
fn test_alloc_only_fn_rejects_io() {
    let src = r#"
fn alloc_only(x: f64) -> f64 / alloc {
    print(x);
    outer(x, x)
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
        "alloc-only fn should reject print (IO): {:?}",
        checker.diagnostics.diagnostics
    );
}

// ── EffectSet helpers ───────────────────────────────────────────

#[test]
fn test_effect_set_pure() {
    assert!(EffectSet::PURE.is_pure());
    assert!(!EffectSet::PURE.has(EffectSet::IO));
    assert!(EffectSet::PURE.is_nogc_safe());
}

#[test]
fn test_effect_set_io() {
    let io = EffectSet::new(EffectSet::IO);
    assert!(!io.is_pure());
    assert!(io.has(EffectSet::IO));
    assert!(!io.has(EffectSet::ALLOC));
}

#[test]
fn test_effect_set_combined() {
    let io_alloc = EffectSet::new(EffectSet::IO | EffectSet::ALLOC);
    assert!(io_alloc.has(EffectSet::IO));
    assert!(io_alloc.has(EffectSet::ALLOC));
    assert!(!io_alloc.has(EffectSet::GC));
    assert!(!io_alloc.is_pure());
}

#[test]
fn test_effect_set_gc_not_nogc_safe() {
    let gc = EffectSet::new(EffectSet::GC);
    assert!(!gc.is_nogc_safe());
}

// ── Effect annotation on impl methods ───────────────────────────

#[test]
fn test_impl_method_effect_annotation() {
    let src = r#"
struct Counter { value: i64 }
impl Counter {
    fn get(self: Counter) -> i64 / pure {
        self.value
    }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    // Check that the method has the effect annotation
    if let cjc_ast::DeclKind::Impl(ref impl_decl) = program.declarations[1].kind {
        assert_eq!(
            impl_decl.methods[0].effect_annotation,
            Some(vec!["pure".to_string()])
        );
    } else {
        panic!("expected ImplDecl");
    }
}

// ── Execution: effect annotations don't affect runtime ──────────

#[test]
fn test_effect_annotated_fn_executes() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 / pure {
    a + b
}
print(add(1, 2));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok(), "execution failed: {:?}", result);
    assert_eq!(interp.output, vec!["3"]);
}

#[test]
fn test_io_annotated_fn_executes() {
    let src = r#"
fn greet(x: i64) / io {
    print(x);
}
greet(42);
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok(), "execution failed: {:?}", result);
    assert_eq!(interp.output, vec!["42"]);
}

// ── Parity: eval vs MIR-exec with effect annotations ───────────

#[test]
fn test_effect_parity() {
    let src = r#"
fn double(x: i64) -> i64 / pure {
    x * 2
}
print(double(21));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    // Eval
    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program);
    let eval_output = interp.output.clone();

    // MIR-exec
    let (_, mir_exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let mir_output = mir_exec.output.clone();

    assert_eq!(eval_output, mir_output, "parity mismatch");
}

// ── Effect registry smoke test ──────────────────────────────────

#[test]
fn test_effect_registry_print_is_io() {
    let registry = cjc_types::effect_registry::builtin_effects();
    let print_effects = registry.get("print").unwrap();
    assert!(print_effects.has(EffectSet::IO));
    assert!(!print_effects.is_pure());
}

#[test]
fn test_effect_registry_sqrt_is_pure() {
    let registry = cjc_types::effect_registry::builtin_effects();
    let sqrt_effects = registry.get("sqrt").unwrap();
    assert!(sqrt_effects.is_pure());
}

#[test]
fn test_effect_registry_tensor_randn_is_nondet() {
    let registry = cjc_types::effect_registry::builtin_effects();
    let randn_effects = registry.get("Tensor.randn").unwrap();
    assert!(randn_effects.has(EffectSet::NONDET));
}
