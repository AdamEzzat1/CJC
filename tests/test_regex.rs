//! CJC Test Suite — First-Class Regex Integration Tests
//!
//! Tests cover:
//!   - Lexer: `/pattern/flags` regex literals, `~=` and `!~` operators, context-sensitive `/`
//!   - Parser: regex literal AST nodes, match/not-match binary ops
//!   - AST eval: regex matching on String, ByteSlice, StrView
//!   - MIR-exec parity: identical results across both pipelines
//!   - Regex engine: pattern matching correctness (via cjc-regex + integration)

use cjc_ast::*;
use cjc_eval::Interpreter;
use cjc_lexer::{Lexer, TokenKind};
use cjc_parser::parse_source;
use cjc_runtime::Value;

// ═══════════════════════════════════════════════════════════════════════
// Helper constructors
// ═══════════════════════════════════════════════════════════════════════

fn span() -> Span { Span::dummy() }
fn string_expr(s: &str) -> Expr { Expr { kind: ExprKind::StringLit(s.to_string()), span: span() } }
fn byte_string_expr(bytes: &[u8]) -> Expr {
    Expr { kind: ExprKind::ByteStringLit(bytes.to_vec()), span: span() }
}

fn regex_expr(pattern: &str, flags: &str) -> Expr {
    Expr { kind: ExprKind::RegexLit { pattern: pattern.to_string(), flags: flags.to_string() }, span: span() }
}

fn binary(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr { kind: ExprKind::Binary { op, left: Box::new(left), right: Box::new(right) }, span: span() }
}

fn eval_expr_val(expr: &Expr) -> Value {
    let mut interp = Interpreter::new(42);
    interp.eval_expr(expr).unwrap()
}

/// Parse and execute source via AST eval, return print output lines.
fn eval_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors:\n{}", diag.render_all(src, "<test>"));
    let mut interp = Interpreter::new(42);
    match interp.exec(&program) {
        Ok(_) => {}
        Err(e) => panic!("eval error: {:?}", e),
    }
    interp.output.clone()
}

/// Parse and execute source via MIR-exec, return print output lines.
fn mir_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors:\n{}", diag.render_all(src, "<test>"));
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("MIR exec failed");
    executor.output.clone()
}

/// Assert AST eval and MIR exec produce identical print output.
fn assert_parity(src: &str) {
    let ast_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(
        ast_out, mir_out,
        "Parity failure for: {}\n  AST output: {:?}\n  MIR output: {:?}",
        src.trim(), ast_out, mir_out
    );
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 1: Lexer Tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_lex_regex_basic() {
    let (tokens, diag) = Lexer::new("/hello/").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::RegexLit);
    assert_eq!(tokens[0].text, "hello"); // no NUL when flags empty
}

#[test]
fn test_lex_regex_with_flags() {
    let (tokens, diag) = Lexer::new("/abc/im").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::RegexLit);
    assert_eq!(tokens[0].text, "abc\0im");
}

#[test]
fn test_lex_regex_with_all_flags() {
    let (tokens, diag) = Lexer::new("/x/gimsx").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::RegexLit);
    assert_eq!(tokens[0].text, "x\0gimsx");
}

#[test]
fn test_lex_regex_with_escapes() {
    let (tokens, diag) = Lexer::new("/a\\/b/").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::RegexLit);
    assert_eq!(tokens[0].text, "a\\/b"); // no NUL when flags empty
}

#[test]
fn test_lex_regex_with_special_chars() {
    let (tokens, diag) = Lexer::new("/\\d+\\.\\w*/").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::RegexLit);
    assert_eq!(tokens[0].text, "\\d+\\.\\w*"); // no NUL when flags empty
}

#[test]
fn test_lex_tilde_eq() {
    let (tokens, diag) = Lexer::new("~=").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::TildeEq);
}

#[test]
fn test_lex_bang_tilde() {
    let (tokens, diag) = Lexer::new("!~").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::BangTilde);
}

#[test]
fn test_lex_tilde_standalone() {
    let (tokens, diag) = Lexer::new("~").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::Tilde);
}

#[test]
fn test_lex_slash_as_division_after_int() {
    let (tokens, diag) = Lexer::new("42 / 7").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::IntLit);
    assert_eq!(tokens[1].kind, TokenKind::Slash);
    assert_eq!(tokens[2].kind, TokenKind::IntLit);
}

#[test]
fn test_lex_slash_as_division_after_ident() {
    let (tokens, diag) = Lexer::new("x / y").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::Ident);
    assert_eq!(tokens[1].kind, TokenKind::Slash);
    assert_eq!(tokens[2].kind, TokenKind::Ident);
}

#[test]
fn test_lex_slash_as_division_after_rparen() {
    let (tokens, diag) = Lexer::new("(x) / 2").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[2].kind, TokenKind::RParen);
    assert_eq!(tokens[3].kind, TokenKind::Slash);
}

#[test]
fn test_lex_regex_after_eq() {
    let (tokens, diag) = Lexer::new("let x = /hello/i").tokenize();
    assert!(!diag.has_errors());
    let regex_tok = tokens.iter().find(|t| t.kind == TokenKind::RegexLit);
    assert!(regex_tok.is_some(), "Expected RegexLit token, got: {:?}",
        tokens.iter().map(|t| t.kind).collect::<Vec<_>>());
    assert_eq!(regex_tok.unwrap().text, "hello\0i");
}

#[test]
fn test_lex_regex_after_tilde_eq() {
    let (tokens, diag) = Lexer::new("x ~= /abc/").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::Ident);
    assert_eq!(tokens[1].kind, TokenKind::TildeEq);
    assert_eq!(tokens[2].kind, TokenKind::RegexLit);
}

#[test]
fn test_lex_operators_not_broken() {
    let (tokens, diag) = Lexer::new("+ - * / % == != < > <= >= && || ! = |> -> =>").tokenize();
    assert!(!diag.has_errors(), "Lex errors: {:?}", diag.diagnostics);
    let kinds: Vec<TokenKind> = tokens.iter()
        .filter(|t| t.kind != TokenKind::Eof)
        .map(|t| t.kind).collect();
    assert_eq!(kinds, vec![
        TokenKind::Plus, TokenKind::Minus, TokenKind::Star, TokenKind::Slash,
        TokenKind::Percent, TokenKind::EqEq, TokenKind::BangEq, TokenKind::Lt,
        TokenKind::Gt, TokenKind::LtEq, TokenKind::GtEq, TokenKind::AmpAmp,
        TokenKind::PipePipe, TokenKind::Bang, TokenKind::Eq, TokenKind::PipeGt,
        TokenKind::Arrow, TokenKind::FatArrow,
    ]);
}

#[test]
fn test_lex_slash_with_space_after() {
    let (tokens, diag) = Lexer::new("/ foo").tokenize();
    assert!(!diag.has_errors());
    assert_eq!(tokens[0].kind, TokenKind::Slash);
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 2: Parser Tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_parse_regex_literal() {
    let src = "let r = /hello/i;";
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors: {:?}", diag.diagnostics);
    let found = program.declarations.iter().any(|d| {
        if let DeclKind::Let(ref ls) = d.kind {
            if let ExprKind::RegexLit { ref pattern, ref flags } = ls.init.kind {
                return pattern == "hello" && flags == "i";
            }
        }
        false
    });
    assert!(found, "Expected RegexLit(hello, i) in program");
}

#[test]
fn test_parse_match_op() {
    let src = r#""hello" ~= /ell/;"#;
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors: {:?}", diag.diagnostics);
    let found = program.declarations.iter().any(|d| {
        if let DeclKind::Stmt(Stmt { kind: StmtKind::Expr(ref e), .. }) = d.kind {
            if let ExprKind::Binary { op: BinOp::Match, .. } = e.kind {
                return true;
            }
        }
        false
    });
    assert!(found, "Expected Binary Match op");
}

#[test]
fn test_parse_not_match_op() {
    let src = r#""hello" !~ /xyz/;"#;
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors: {:?}", diag.diagnostics);
    let found = program.declarations.iter().any(|d| {
        if let DeclKind::Stmt(Stmt { kind: StmtKind::Expr(ref e), .. }) = d.kind {
            if let ExprKind::Binary { op: BinOp::NotMatch, .. } = e.kind {
                return true;
            }
        }
        false
    });
    assert!(found, "Expected Binary NotMatch op");
}

#[test]
fn test_parse_regex_in_let() {
    let src = r#"let matched = "test123" ~= /\d+/;"#;
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors: {:?}", diag.diagnostics);
    let found = program.declarations.iter().any(|d| {
        if let DeclKind::Let(ref ls) = d.kind {
            if ls.name.name == "matched" {
                if let ExprKind::Binary { op: BinOp::Match, .. } = ls.init.kind {
                    return true;
                }
            }
        }
        false
    });
    assert!(found, "Expected let matched = ... ~= ...");
}

#[test]
fn test_parse_regex_precedence() {
    let src = r#""x" ~= /x/ == true;"#;
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors: {:?}", diag.diagnostics);
    // Should parse as ("x" ~= /x/) == true due to left-associativity
    let found = program.declarations.iter().any(|d| {
        if let DeclKind::Stmt(Stmt { kind: StmtKind::Expr(ref e), .. }) = d.kind {
            if let ExprKind::Binary { op: BinOp::Eq, ref left, .. } = e.kind {
                if let ExprKind::Binary { op: BinOp::Match, .. } = left.kind {
                    return true;
                }
            }
        }
        false
    });
    assert!(found, "Expected (... ~= ...) == true");
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 3: AST Eval Tests — Direct AST construction
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_eval_regex_literal_value() {
    let expr = regex_expr("hello", "i");
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Regex { ref pattern, ref flags } if pattern == "hello" && flags == "i"),
        "Expected Regex(hello, i), got {}", val);
}

#[test]
fn test_eval_regex_display() {
    let val = eval_expr_val(&regex_expr("abc", "gm"));
    assert_eq!(format!("{}", val), "/abc/gm");
}

#[test]
fn test_eval_string_match_regex() {
    let expr = binary(BinOp::Match, string_expr("hello world"), regex_expr("hello", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(true)));
}

#[test]
fn test_eval_string_no_match_regex() {
    let expr = binary(BinOp::Match, string_expr("hello world"), regex_expr("xyz", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(false)));
}

#[test]
fn test_eval_string_not_match_regex() {
    let expr = binary(BinOp::NotMatch, string_expr("hello world"), regex_expr("xyz", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(true)));
}

#[test]
fn test_eval_string_not_match_regex_false() {
    let expr = binary(BinOp::NotMatch, string_expr("hello world"), regex_expr("world", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(false)));
}

#[test]
fn test_eval_byteslice_match_regex() {
    let expr = binary(BinOp::Match, byte_string_expr(b"hello123"), regex_expr("\\d+", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(true)));
}

#[test]
fn test_eval_byteslice_no_match_regex() {
    let expr = binary(BinOp::Match, byte_string_expr(b"hello"), regex_expr("\\d+", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(false)));
}

#[test]
fn test_eval_regex_case_insensitive() {
    let expr = binary(BinOp::Match, string_expr("Hello World"), regex_expr("hello", "i"));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(true)));
}

#[test]
fn test_eval_regex_case_sensitive_fail() {
    let expr = binary(BinOp::Match, string_expr("Hello World"), regex_expr("hello", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(false)));
}

#[test]
fn test_eval_regex_anchor_start() {
    let yes = binary(BinOp::Match, string_expr("abc"), regex_expr("^abc", ""));
    let no = binary(BinOp::Match, string_expr("xabc"), regex_expr("^abc", ""));
    assert!(matches!(eval_expr_val(&yes), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

#[test]
fn test_eval_regex_anchor_end() {
    let yes = binary(BinOp::Match, string_expr("abc"), regex_expr("abc$", ""));
    let no = binary(BinOp::Match, string_expr("abcx"), regex_expr("abc$", ""));
    assert!(matches!(eval_expr_val(&yes), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

#[test]
fn test_eval_regex_dot_any() {
    let expr = binary(BinOp::Match, string_expr("a1b"), regex_expr("a.b", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(true)));
}

#[test]
fn test_eval_regex_alternation() {
    let yes1 = binary(BinOp::Match, string_expr("cat"), regex_expr("cat|dog", ""));
    let yes2 = binary(BinOp::Match, string_expr("dog"), regex_expr("cat|dog", ""));
    let no = binary(BinOp::Match, string_expr("bird"), regex_expr("cat|dog", ""));
    assert!(matches!(eval_expr_val(&yes1), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&yes2), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

#[test]
fn test_eval_regex_char_class() {
    let yes = binary(BinOp::Match, string_expr("abc"), regex_expr("[abc]", ""));
    let no = binary(BinOp::Match, string_expr("xyz"), regex_expr("[abc]", ""));
    assert!(matches!(eval_expr_val(&yes), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

#[test]
fn test_eval_regex_negated_class() {
    let yes = binary(BinOp::Match, string_expr("x"), regex_expr("[^abc]", ""));
    let no = binary(BinOp::Match, string_expr("a"), regex_expr("^[^abc]$", ""));
    assert!(matches!(eval_expr_val(&yes), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

#[test]
fn test_eval_regex_quantifiers() {
    let star = binary(BinOp::Match, string_expr("aaa"), regex_expr("^a*$", ""));
    let plus = binary(BinOp::Match, string_expr("aaa"), regex_expr("^a+$", ""));
    let quest = binary(BinOp::Match, string_expr("a"), regex_expr("^ab?$", ""));
    assert!(matches!(eval_expr_val(&star), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&plus), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&quest), Value::Bool(true)));
}

#[test]
fn test_eval_regex_digit_class() {
    let yes = binary(BinOp::Match, string_expr("abc123"), regex_expr("\\d+", ""));
    let no = binary(BinOp::Match, string_expr("abc"), regex_expr("^\\d+$", ""));
    assert!(matches!(eval_expr_val(&yes), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

#[test]
fn test_eval_regex_word_class() {
    let yes = binary(BinOp::Match, string_expr("hello_world"), regex_expr("^\\w+$", ""));
    let no = binary(BinOp::Match, string_expr("hello world"), regex_expr("^\\w+$", ""));
    assert!(matches!(eval_expr_val(&yes), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 4: Full Pipeline Tests (parse_source → eval, using print)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_regex_match_string() {
    let out = eval_output(r#"print("hello world" ~= /world/);"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_regex_no_match_string() {
    let out = eval_output(r#"print("hello world" ~= /xyz/);"#);
    assert_eq!(out, vec!["false"]);
}

#[test]
fn test_pipeline_regex_not_match() {
    let out = eval_output(r#"print("hello world" !~ /xyz/);"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_regex_case_insensitive() {
    let out = eval_output(r#"print("Hello World" ~= /hello/i);"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_regex_digit_pattern() {
    let out = eval_output(r#"print("order-42" ~= /\d+/);"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_regex_anchored() {
    let out = eval_output(r#"print("abc" ~= /^abc$/);"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_regex_complex_pattern() {
    let out = eval_output(r#"print("2024-01-15" ~= /\d\d\d\d-\d\d-\d\d/);"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_regex_let_binding() {
    let out = eval_output(r#"
        let r = /\d+/;
        let s = "abc123";
        print(s ~= r);
    "#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_regex_in_if() {
    let out = eval_output(r#"
        let s = "error: bad input";
        if s ~= /^error/ {
            print(42);
        } else {
            print(0);
        }
    "#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn test_pipeline_regex_in_function() {
    let out = eval_output(r#"
        fn is_email(s: String) -> Bool {
            return s ~= /\w+@\w+\.\w+/;
        }
        print(is_email("user@example.com"));
    "#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_regex_not_match_in_function() {
    let out = eval_output(r#"
        fn is_not_numeric(s: String) -> Bool {
            return s !~ /^\d+$/;
        }
        print(is_not_numeric("hello"));
    "#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn test_pipeline_division_still_works() {
    let out = eval_output("print(42 / 7);");
    assert_eq!(out, vec!["6"]);
}

#[test]
fn test_pipeline_division_float() {
    let out = eval_output("print(10.0 / 4.0);");
    assert_eq!(out, vec!["2.5"]);
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 5: MIR-exec Parity Tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_parity_regex_match() {
    assert_parity(r#"print("hello world" ~= /world/);"#);
}

#[test]
fn test_parity_regex_no_match() {
    assert_parity(r#"print("hello" ~= /xyz/);"#);
}

#[test]
fn test_parity_regex_not_match() {
    assert_parity(r#"print("hello" !~ /xyz/);"#);
}

#[test]
fn test_parity_regex_case_insensitive() {
    assert_parity(r#"print("Hello" ~= /hello/i);"#);
}

#[test]
fn test_parity_regex_digit_match() {
    assert_parity(r#"print("abc123" ~= /\d+/);"#);
}

#[test]
fn test_parity_regex_anchored_match() {
    assert_parity(r#"print("abc" ~= /^abc$/);"#);
}

#[test]
fn test_parity_regex_alternation() {
    assert_parity(r#"print("dog" ~= /cat|dog/);"#);
}

#[test]
fn test_parity_regex_let_and_use() {
    assert_parity(r#"
        let pat = /\w+/;
        let s = "hello";
        print(s ~= pat);
    "#);
}

#[test]
fn test_parity_regex_in_if() {
    assert_parity(r#"
        let input = "ERROR: something";
        if input ~= /^ERROR/ {
            print(1);
        } else {
            print(0);
        }
    "#);
}

#[test]
fn test_parity_regex_in_function() {
    assert_parity(r#"
        fn has_digits(s: String) -> Bool {
            return s ~= /\d/;
        }
        print(has_digits("abc123"));
    "#);
}

#[test]
fn test_parity_division_not_broken() {
    assert_parity("print(100 / 10);");
}

#[test]
fn test_parity_regex_word_boundary() {
    assert_parity(r#"print("hello world" ~= /\bworld\b/);"#);
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 6: Edge Cases and Error Handling
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_eval_regex_empty_pattern() {
    let expr = binary(BinOp::Match, string_expr("anything"), regex_expr("", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(true)));
}

#[test]
fn test_eval_regex_empty_string() {
    let yes = binary(BinOp::Match, string_expr(""), regex_expr("^$", ""));
    let no = binary(BinOp::Match, string_expr(""), regex_expr("\\d+", ""));
    assert!(matches!(eval_expr_val(&yes), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

#[test]
fn test_eval_regex_special_bytes() {
    let bytes = vec![0x80, 0x81, 0x42, 0x43]; // non-UTF8, then BC
    let expr = binary(BinOp::Match, byte_string_expr(&bytes), regex_expr("BC", ""));
    assert!(matches!(eval_expr_val(&expr), Value::Bool(true)));
}

#[test]
fn test_eval_regex_escaped_dot() {
    let yes = binary(BinOp::Match, string_expr("a.b"), regex_expr("a\\.b", ""));
    let no = binary(BinOp::Match, string_expr("axb"), regex_expr("a\\.b", ""));
    assert!(matches!(eval_expr_val(&yes), Value::Bool(true)));
    assert!(matches!(eval_expr_val(&no), Value::Bool(false)));
}

#[test]
fn test_eval_regex_type_name() {
    let val = eval_expr_val(&regex_expr("test", ""));
    assert_eq!(val.type_name(), "Regex");
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 7: Regex Engine Direct Tests (via cjc_regex crate)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_cjc_regex_is_match() {
    assert!(cjc_regex::is_match("hello", "", b"say hello world"));
    assert!(!cjc_regex::is_match("^hello$", "", b"say hello world"));
    assert!(cjc_regex::is_match("^hello$", "", b"hello"));
}

#[test]
fn test_cjc_regex_find() {
    let m = cjc_regex::find("\\d+", "", b"abc123def456");
    assert_eq!(m, Some((3, 6)));
}

#[test]
fn test_cjc_regex_find_all() {
    let matches = cjc_regex::find_all("\\d+", "", b"abc123def456ghi");
    assert_eq!(matches, vec![(3, 6), (9, 12)]);
}

#[test]
fn test_cjc_regex_split() {
    let ranges = cjc_regex::split(",\\s*", "", b"a, b,c,  d");
    let hay = b"a, b,c,  d";
    let parts: Vec<&[u8]> = ranges.iter().map(|(s, e)| &hay[*s..*e]).collect();
    assert_eq!(parts, vec![b"a".as_slice(), b"b", b"c", b"d"]);
}

#[test]
fn test_cjc_regex_case_insensitive_flag() {
    assert!(cjc_regex::is_match("HELLO", "i", b"hello"));
    assert!(!cjc_regex::is_match("HELLO", "", b"hello"));
}

#[test]
fn test_cjc_regex_char_range() {
    assert!(cjc_regex::is_match("^[a-z]+$", "", b"hello"));
    assert!(!cjc_regex::is_match("^[a-z]+$", "", b"Hello"));
}

#[test]
fn test_cjc_regex_hex_escape() {
    assert!(cjc_regex::is_match("\\x41", "", b"A"));
    assert!(!cjc_regex::is_match("\\x41", "", b"b"));
}

#[test]
fn test_cjc_regex_quantifier_star_empty() {
    assert!(cjc_regex::is_match("^a*$", "", b""));
    assert!(cjc_regex::is_match("^a*$", "", b"aaa"));
}
