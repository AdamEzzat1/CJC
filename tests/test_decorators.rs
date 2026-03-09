// CJC Test Suite — Decorators (@memoize, @trace)
// Tests lexer, parser, and runtime behavior for decorator syntax.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_eval(src: &str) -> Vec<String> {
    let (prog, diag) = cjc_parser::parse_source(src);
    if diag.count() > 0 {
        panic!("parse errors:\n{}", diag.render_all(src, "<test>"));
    }
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&prog).unwrap();
    interp.output.clone()
}

fn run_mir(src: &str) -> Vec<String> {
    let (prog, diag) = cjc_parser::parse_source(src);
    if diag.count() > 0 {
        panic!("parse errors:\n{}", diag.render_all(src, "<test>"));
    }
    let (_val, exec) = cjc_mir_exec::run_program_with_executor(&prog, 42).unwrap();
    exec.output.clone()
}

// ---------------------------------------------------------------------------
// Lexer tests
// ---------------------------------------------------------------------------

#[test]
fn lex_at_token() {
    let (tokens, diag) = cjc_lexer::Lexer::new("@memoize").tokenize();
    assert_eq!(diag.count(), 0, "no diagnostics expected");
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::At);
    assert_eq!(tokens[0].text, "@");
    assert_eq!(tokens[1].kind, cjc_lexer::TokenKind::Ident);
    assert_eq!(tokens[1].text, "memoize");
}

#[test]
fn lex_at_with_args() {
    let (tokens, diag) = cjc_lexer::Lexer::new("@log(42)").tokenize();
    assert_eq!(diag.count(), 0);
    assert_eq!(tokens[0].kind, cjc_lexer::TokenKind::At);
    assert_eq!(tokens[1].kind, cjc_lexer::TokenKind::Ident);
    assert_eq!(tokens[1].text, "log");
    assert_eq!(tokens[2].kind, cjc_lexer::TokenKind::LParen);
    assert_eq!(tokens[3].kind, cjc_lexer::TokenKind::IntLit);
    assert_eq!(tokens[4].kind, cjc_lexer::TokenKind::RParen);
}

// ---------------------------------------------------------------------------
// Parser tests
// ---------------------------------------------------------------------------

#[test]
fn parse_single_decorator() {
    let src = "
@memoize
fn double(x: i64) -> i64 {
    x * 2
}
";
    let (prog, diag) = cjc_parser::parse_source(src);
    if diag.count() > 0 {
        eprintln!("DIAGNOSTICS:\n{}", diag.render_all(src, "<test>"));
    }
    assert_eq!(diag.count(), 0, "no diagnostics");
    assert_eq!(prog.declarations.len(), 1);
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => {
            assert_eq!(f.decorators.len(), 1);
            assert_eq!(f.decorators[0].name.name, "memoize");
            assert!(f.decorators[0].args.is_empty());
        }
        _ => panic!("expected FnDecl"),
    }
}

#[test]
fn parse_multiple_decorators() {
    let src = "
@trace
@memoize
fn add(a: i64, b: i64) -> i64 {
    a + b
}
";
    let (prog, diag) = cjc_parser::parse_source(src);
    assert_eq!(diag.count(), 0, "no diagnostics");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => {
            assert_eq!(f.decorators.len(), 2);
            assert_eq!(f.decorators[0].name.name, "trace");
            assert_eq!(f.decorators[1].name.name, "memoize");
        }
        _ => panic!("expected FnDecl"),
    }
}

#[test]
fn parse_decorator_with_args() {
    let src = "
@retry(3)
fn fetch(url: String) -> String {
    url
}
";
    let (prog, diag) = cjc_parser::parse_source(src);
    assert_eq!(diag.count(), 0, "no diagnostics");
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => {
            assert_eq!(f.decorators.len(), 1);
            assert_eq!(f.decorators[0].name.name, "retry");
            assert_eq!(f.decorators[0].args.len(), 1);
        }
        _ => panic!("expected FnDecl"),
    }
}

#[test]
fn parse_no_decorator_fn_unchanged() {
    let src = "fn add(a: i64, b: i64) -> i64 { a + b }";
    let (prog, diag) = cjc_parser::parse_source(src);
    assert_eq!(diag.count(), 0);
    match &prog.declarations[0].kind {
        cjc_ast::DeclKind::Fn(f) => {
            assert!(f.decorators.is_empty());
        }
        _ => panic!("expected FnDecl"),
    }
}

// ---------------------------------------------------------------------------
// Runtime: @memoize (eval)
// ---------------------------------------------------------------------------

#[test]
fn eval_memoize_caches_result() {
    let src = r#"
@memoize
fn double(x: i64) -> i64 {
    x * 2
}
let a = double(5);
let b = double(5);
print(a + b);
"#;
    let output = run_eval(src);
    assert!(output.iter().any(|s| s.contains("20")), "expected 20 in output, got {:?}", output);
}

#[test]
fn eval_trace_produces_output() {
    let src = r#"
@trace
fn square(x: i64) -> i64 {
    x * x
}
square(3);
"#;
    let output = run_eval(src);
    // Check trace output — should have entry and exit messages
    assert!(output.len() >= 2, "expected at least 2 trace messages, got {:?}", output);
    assert!(output[0].contains("[trace]") && output[0].contains("enter"),
            "expected entry trace, got: {}", output[0]);
    assert!(output[1].contains("[trace]") && output[1].contains("=>"),
            "expected exit trace, got: {}", output[1]);
}

#[test]
fn eval_memoize_and_trace_combined() {
    let src = r#"
@trace
@memoize
fn add(a: i64, b: i64) -> i64 {
    a + b
}
let x = add(1, 2);
let y = add(1, 2);
print(x + y);
"#;
    let output = run_eval(src);
    // First call: trace enter + trace exit (2 messages)
    // Second call: trace cached (1 message)
    // Then print output
    let trace_msgs: Vec<_> = output.iter().filter(|s| s.contains("[trace]")).collect();
    assert!(trace_msgs.len() >= 3, "expected 3+ trace messages, got {:?}", trace_msgs);
    assert!(trace_msgs[2].contains("cached"), "expected cache hit trace, got: {}", trace_msgs[2]);
}

// ---------------------------------------------------------------------------
// Runtime: @memoize (MIR-exec)
// ---------------------------------------------------------------------------

#[test]
fn mir_memoize_caches_result() {
    let src = r#"
@memoize
fn double(x: i64) -> i64 {
    x * 2
}
let a = double(5);
let b = double(5);
print(a + b);
"#;
    let output = run_mir(src);
    assert!(output.iter().any(|s| s.contains("20")), "expected 20 in output, got {:?}", output);
}

#[test]
fn mir_trace_produces_output() {
    let src = r#"
@trace
fn square(x: i64) -> i64 {
    x * x
}
square(3);
"#;
    let output = run_mir(src);
    assert!(output.len() >= 2, "expected at least 2 trace messages, got {:?}", output);
    assert!(output[0].contains("[trace]") && output[0].contains("enter"),
            "expected entry trace, got: {}", output[0]);
}

// ---------------------------------------------------------------------------
// Parity: eval vs MIR-exec
// ---------------------------------------------------------------------------

#[test]
fn parity_decorator_memoize_fib() {
    let src = r#"
@memoize
fn fib(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    fib(n - 1) + fib(n - 2)
}
print(fib(10));
"#;
    let eval_out = run_eval(src);
    let mir_out = run_mir(src);

    // Filter out trace messages, just compare print output
    let eval_prints: Vec<_> = eval_out.iter().filter(|s| !s.contains("[trace]")).collect();
    let mir_prints: Vec<_> = mir_out.iter().filter(|s| !s.contains("[trace]")).collect();

    assert_eq!(eval_prints, mir_prints, "parity: eval={:?} mir={:?}", eval_prints, mir_prints);
    assert!(eval_prints.iter().any(|s| s.contains("55")), "expected 55 in output, got {:?}", eval_prints);
}
