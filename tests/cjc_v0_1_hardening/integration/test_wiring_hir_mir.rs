//! HIR → MIR wiring verification.
//! Confirms that AST → HIR → MIR lowering handles all major constructs.

/// Verify that a source program successfully lowers to HIR.
fn check_hir_lowers(label: &str, src: &str) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "[{label}] Parse errors: {:?}",
        diags.diagnostics
    );
    let mut lowerer = cjc_hir::AstLowering::new();
    let hir_result = lowerer.lower_program(&program);
    assert!(
        !hir_result.items.is_empty() || program.declarations.is_empty(),
        "[{label}] HIR lowering produced no items"
    );
}

/// Verify that a source program successfully executes via MIR.
fn check_mir_runs(label: &str, src: &str) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "[{label}] Parse errors: {:?}", diags.diagnostics);
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    assert!(result.is_ok(), "[{label}] MIR-exec failed: {:?}", result.err());
}

// ============================================================
// AST → HIR lowering tests
// ============================================================

#[test]
fn hir_function_def() {
    check_hir_lowers("fn def", "fn foo(x: i64) -> i64 { x + 1 }");
}

#[test]
fn hir_struct_def() {
    check_hir_lowers("struct", "struct Point { x: f64, y: f64 }");
}

#[test]
fn hir_let_binding() {
    check_hir_lowers("let", "fn main() { let x: i64 = 42; }");
}

#[test]
fn hir_if_else() {
    check_hir_lowers("if-else", "fn main() -> i64 { if true { 1 } else { 2 } }");
}

#[test]
fn hir_while_loop() {
    check_hir_lowers("while", r#"
fn main() {
    let mut i: i64 = 0;
    while i < 10 { i = i + 1; }
}
"#);
}

#[test]
fn hir_for_loop() {
    check_hir_lowers("for", r#"
fn main() {
    let mut s: i64 = 0;
    for i in 0..10 { s = s + i; }
}
"#);
}

#[test]
fn hir_closure() {
    check_hir_lowers("closure", r#"
fn main() {
    let f = |x: i64| x + 1;
}
"#);
}

#[test]
fn hir_match_expression() {
    check_hir_lowers("match", r#"
fn main() -> i64 {
    match 1 {
        1 => 10,
        _ => 0,
    }
}
"#);
}

// ============================================================
// MIR execution tests (full pipeline)
// ============================================================

#[test]
fn mir_full_pipeline_arithmetic() {
    check_mir_runs("arith", "fn main() -> i64 { 1 + 2 * 3 }");
}

#[test]
fn mir_full_pipeline_function_call() {
    check_mir_runs("fn call", r#"
fn f(x: i64) -> i64 { x * 2 }
fn main() -> i64 { f(21) }
"#);
}

#[test]
fn mir_full_pipeline_recursion() {
    check_mir_runs("recursion", r#"
fn fib(n: i64) -> i64 {
    if n <= 1 { return n; }
    fib(n - 1) + fib(n - 2)
}
fn main() -> i64 { fib(10) }
"#);
}

#[test]
fn mir_full_pipeline_nested_loops() {
    check_mir_runs("nested loops", r#"
fn main() -> i64 {
    let mut total: i64 = 0;
    for i in 0..5 {
        for j in 0..5 {
            total = total + 1;
        }
    }
    total
}
"#);
}

#[test]
fn mir_full_pipeline_closure_with_capture() {
    check_mir_runs("closure capture", r#"
fn main() -> i64 {
    let x: i64 = 10;
    let f = |y: i64| x + y;
    f(32)
}
"#);
}

#[test]
fn mir_full_pipeline_struct() {
    check_mir_runs("struct", r#"
struct Pair { a: i64, b: i64 }
fn main() -> i64 {
    let p = Pair { a: 3, b: 4 };
    p.a + p.b
}
"#);
}

#[test]
fn mir_full_pipeline_if_expression() {
    check_mir_runs("if expr", r#"
fn main() -> i64 {
    let x = if true { 42 } else { 0 };
    x
}
"#);
}

#[test]
fn mir_full_pipeline_match() {
    check_mir_runs("match", r#"
fn main() -> i64 {
    let x: i64 = 3;
    match x {
        1 => 10,
        2 => 20,
        3 => 30,
        _ => 0,
    }
}
"#);
}
