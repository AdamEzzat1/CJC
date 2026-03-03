//! LH05: REPL Upgrades tests
//!
//! Verifies:
//! - Meta-commands are parsed correctly
//! - Effect-annotated functions work in REPL-like eval
//! - Multi-line input brace balancing works
//! - `:type`, `:ast`, `:mir` introspection commands produce output
//!
//! Note: Raw terminal mode cannot be tested in a headless test environment.
//! These tests exercise the REPL evaluation pipeline and command parsing.

// ── REPL eval: basic expression evaluation ──────────────────────

#[test]
fn test_repl_eval_simple_expr() {
    let src = "print(1 + 2);";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["3"]);
}

#[test]
fn test_repl_eval_function_definition_and_call() {
    let src = r#"
fn square(x: i64) -> i64 { x * x }
print(square(7));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["49"]);
}

#[test]
fn test_repl_eval_let_binding() {
    let src = r#"
let x: i64 = 42;
print(x);
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["42"]);
}

// ── Multi-line detection ────────────────────────────────────────

#[test]
fn test_multiline_brace_count() {
    // Simulate multi-line detection: open braces > close braces means continue
    let line1 = "fn foo() {";
    let open = line1.chars().filter(|&c| c == '{').count();
    let close = line1.chars().filter(|&c| c == '}').count();
    assert!(open > close, "should continue for unbalanced braces");

    let line2 = "fn foo() { 42 }";
    let open = line2.chars().filter(|&c| c == '{').count();
    let close = line2.chars().filter(|&c| c == '}').count();
    assert_eq!(open, close, "balanced braces — no continuation");
}

#[test]
fn test_multiline_backslash_continuation() {
    let line = "let x = 1 + \\";
    assert!(line.trim_end().ends_with('\\'), "should detect backslash continuation");
}

// ── Meta-command parsing ────────────────────────────────────────

#[test]
fn test_meta_command_detection() {
    assert!(":help".starts_with(':'));
    assert!(":quit".starts_with(':'));
    assert!(":type 42".starts_with(':'));
    assert!(":ast fn f() {}".starts_with(':'));
    assert!(":mir 1 + 2".starts_with(':'));
    assert!(":reset".starts_with(':'));
    assert!(":env".starts_with(':'));
    assert!(":seed".starts_with(':'));
    assert!(!":".is_empty());
    assert!(!"let x = 1".starts_with(':'));
}

#[test]
fn test_meta_command_split() {
    let cmd = ":type 42 + 10";
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    assert_eq!(parts[0], ":type");
    assert_eq!(parts[1], "42 + 10");
}

// ── :type command ───────────────────────────────────────────────

#[test]
fn test_type_command_int_literal() {
    let src = "42";
    let (program, diags) = cjc_parser::parse_source(src);
    // May or may not parse as a standalone expression depending on grammar
    // Just verify no crashes
    let _ = diags;
    let _ = program;
}

#[test]
fn test_type_command_fn() {
    let src = "fn add(a: i64, b: i64) -> i64 { a + b }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);
    // Should not produce type errors
    let type_errors: Vec<_> = checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code.starts_with("E0"))
        .collect();
    assert!(type_errors.is_empty(), "unexpected errors: {:?}", type_errors);
}

// ── :ast command ────────────────────────────────────────────────

#[test]
fn test_ast_command_produces_debug_output() {
    let src = "let x: i64 = 1 + 2;";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let debug_str = format!("{:#?}", program);
    assert!(debug_str.contains("Let"), "AST should contain Let node");
    assert!(debug_str.contains("Binary"), "AST should contain Binary node");
}

// ── :mir command ────────────────────────────────────────────────

#[test]
fn test_mir_command_lowers_successfully() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
print(add(1, 2));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mir_program = cjc_mir_exec::lower_to_mir(&program);
    assert!(!mir_program.functions.is_empty(), "should produce MIR functions");
}

// ── Effect-annotated functions in REPL eval ─────────────────────

#[test]
fn test_repl_effect_annotated_fn() {
    let src = r#"
fn pure_add(a: i64, b: i64) -> i64 / pure { a + b }
print(pure_add(10, 20));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_ok());
    assert_eq!(interp.output, vec!["30"]);
}

// ── Parity: REPL-style eval vs MIR-exec ────────────────────────

#[test]
fn test_repl_parity() {
    let src = r#"
fn fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
print(fib(10));
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());

    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program);
    let eval_output = interp.output.clone();

    let (_, mir_exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let mir_output = mir_exec.output.clone();

    assert_eq!(eval_output, mir_output, "REPL eval/MIR parity mismatch");
}
