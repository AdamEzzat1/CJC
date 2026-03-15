//! Tests for variadic function parameters: `fn f(...args: f64)`
//!
//! Covers: parsing, eval, MIR-exec, parity, and error cases.

// ── Helpers ─────────────────────────────────────────────────────

fn eval_output(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        diags.diagnostics.is_empty(),
        "parse errors: {:?}",
        diags.diagnostics
    );
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        diags.diagnostics.is_empty(),
        "parse errors: {:?}",
        diags.diagnostics
    );
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("mir-exec failed");
    executor.output
}

fn both_output(src: &str) -> (Vec<String>, Vec<String>) {
    (eval_output(src), mir_output(src))
}

// ── Lexer ───────────────────────────────────────────────────────

#[test]
fn lexer_dotdotdot() {
    let (tokens, _) = cjc_lexer::Lexer::new("...").tokenize();
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::DotDotDot);
}

#[test]
fn lexer_dotdot_vs_dotdotdot() {
    let (tokens, _) = cjc_lexer::Lexer::new(".. ...").tokenize();
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::DotDot);
    assert_eq!(tokens[1].kind, cjc_lexer::TokenKind::DotDotDot);
}

// ── Parsing ─────────────────────────────────────────────────────

#[test]
fn parse_variadic_param() {
    let src = "fn sum(...values: f64) -> f64 { 0.0 }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(diags.diagnostics.is_empty());
    let decl = &program.declarations[0];
    if let cjc_ast::DeclKind::Fn(ref f) = decl.kind {
        assert_eq!(f.params.len(), 1);
        assert!(f.params[0].is_variadic);
        assert_eq!(f.params[0].name.name, "values");
    } else {
        panic!("expected FnDecl");
    }
}

#[test]
fn parse_variadic_with_regular_params() {
    let src = "fn format(prefix: String, ...args: Any) -> String { prefix }";
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(diags.diagnostics.is_empty());
    if let cjc_ast::DeclKind::Fn(ref f) = program.declarations[0].kind {
        assert_eq!(f.params.len(), 2);
        assert!(!f.params[0].is_variadic);
        assert!(f.params[1].is_variadic);
    } else {
        panic!("expected FnDecl");
    }
}

#[test]
fn parse_variadic_default_rejected() {
    let src = "fn bad(...args: f64 = 1.0) {}";
    let (_, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.diagnostics.is_empty(),
        "variadic with default should produce parse error"
    );
}

// ── Execution: basic variadic ───────────────────────────────────

#[test]
fn variadic_sum_eval_and_mir() {
    let src = r#"
fn sum(...values: f64) -> f64 {
    let total = 0.0;
    let i = 0;
    while i < len(values) {
        total = total + values[i];
        i = i + 1;
    }
    total
}
print(sum(1.0, 2.0, 3.0, 4.0));
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], "10");
    assert_eq!(mir[0], "10");
}

#[test]
fn variadic_zero_args() {
    let src = r#"
fn count(...items: Any) -> i64 {
    len(items)
}
print(count());
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], "0");
    assert_eq!(mir[0], "0");
}

#[test]
fn variadic_with_leading_params() {
    let src = r#"
fn first_plus_rest(first: f64, ...rest: f64) -> f64 {
    let total = first;
    let i = 0;
    while i < len(rest) {
        total = total + rest[i];
        i = i + 1;
    }
    total
}
print(first_plus_rest(10.0, 1.0, 2.0, 3.0));
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], "16");
    assert_eq!(mir[0], "16");
}

#[test]
fn variadic_single_arg() {
    let src = r#"
fn wrap(...items: i64) -> i64 {
    items[0]
}
print(wrap(42));
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], "42");
    assert_eq!(mir[0], "42");
}

// ── Parity: eval vs MIR-exec ───────────────────────────────────

#[test]
fn variadic_parity_identical() {
    let src = r#"
fn collect(...vals: f64) -> f64 {
    let s = 0.0;
    let i = 0;
    while i < len(vals) {
        s = s + vals[i];
        i = i + 1;
    }
    s
}
print(collect(1.1, 2.2, 3.3));
"#;
    let (ev, mir) = both_output(src);
    assert_eq!(ev[0], mir[0], "eval vs mir-exec parity");
}

// ── Determinism ─────────────────────────────────────────────────

#[test]
fn variadic_determinism() {
    let src = r#"
fn cat(...words: String) -> String {
    let result = "";
    let i = 0;
    while i < len(words) {
        result = result + words[i];
        i = i + 1;
    }
    result
}
print(cat("hello", " ", "world"));
"#;
    let v1 = eval_output(src);
    let v2 = eval_output(src);
    assert_eq!(v1[0], v2[0]);
    assert_eq!(v1[0], "hello world");
}
