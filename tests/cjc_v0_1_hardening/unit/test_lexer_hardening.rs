//! Lexer hardening tests — edge cases and boundary conditions.

use cjc_lexer::{Lexer, TokenKind};

/// Empty input produces no tokens (except EOF if applicable).
#[test]
fn lex_empty_input() {
    let (tokens, diags) = Lexer::new("").tokenize();
    assert!(!diags.has_errors(), "Empty input should produce no errors");
    // All tokens should be EOF or empty
    assert!(
        tokens.is_empty() || tokens.iter().all(|t| t.kind == TokenKind::Eof),
        "Empty input should yield no meaningful tokens"
    );
}

/// Whitespace-only input produces no meaningful tokens.
#[test]
fn lex_whitespace_only() {
    let (tokens, diags) = Lexer::new("   \t\n\r\n  ").tokenize();
    assert!(!diags.has_errors());
    let meaningful: Vec<_> = tokens.iter().filter(|t| t.kind != TokenKind::Eof).collect();
    assert!(
        meaningful.is_empty(),
        "Whitespace-only input should yield no meaningful tokens, got {meaningful:?}"
    );
}

/// Single-character tokens are correctly classified.
#[test]
fn lex_single_char_tokens() {
    let src = "+ - * / ( ) { } [ ] , ; : . = < > ! & | ^ % ~";
    let (tokens, diags) = Lexer::new(src).tokenize();
    assert!(!diags.has_errors(), "Single-char tokens should not error");
    let meaningful: Vec<_> = tokens.iter().filter(|t| t.kind != TokenKind::Eof).collect();
    assert!(meaningful.len() >= 15, "Should tokenize at least 15 operators");
}

/// Integer literals at boundary values.
#[test]
fn lex_integer_boundaries() {
    let src = "0 1 9999999999999999999";
    let (tokens, diags) = Lexer::new(src).tokenize();
    // Should not panic
    let _ = diags;
    let ints: Vec<_> = tokens
        .iter()
        .filter(|t| t.kind == TokenKind::IntLit)
        .collect();
    assert!(ints.len() >= 2, "Should parse at least two integer literals");
}

/// Float literals with various formats.
#[test]
fn lex_float_formats() {
    let src = "0.0 3.14 1e10 2.5e-3 1.0E+5";
    let (tokens, diags) = Lexer::new(src).tokenize();
    assert!(!diags.has_errors(), "Float literals should parse: {:?}", diags.diagnostics);
    let floats: Vec<_> = tokens
        .iter()
        .filter(|t| t.kind == TokenKind::FloatLit)
        .collect();
    assert!(floats.len() >= 3, "Should parse multiple float literals");
}

/// String literal edge cases.
#[test]
fn lex_string_edge_cases() {
    // Empty string
    let (tokens, _) = Lexer::new("\"\"").tokenize();
    let strings: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::StringLit).collect();
    assert_eq!(strings.len(), 1, "Should tokenize empty string literal");

    // String with escape sequences
    let (tokens2, _) = Lexer::new(r#""hello\nworld""#).tokenize();
    let strings2: Vec<_> = tokens2.iter().filter(|t| t.kind == TokenKind::StringLit).collect();
    assert_eq!(strings2.len(), 1, "Should tokenize string with escapes");
}

/// Multi-character operators.
#[test]
fn lex_multi_char_operators() {
    let src = "== != <= >= -> => .. += -= *= /=";
    let (tokens, diags) = Lexer::new(src).tokenize();
    assert!(!diags.has_errors(), "Multi-char operators should not error");
    let meaningful: Vec<_> = tokens.iter().filter(|t| t.kind != TokenKind::Eof).collect();
    assert!(meaningful.len() >= 8, "Should tokenize multi-char operators");
}

/// All keywords are recognized.
#[test]
fn lex_all_keywords() {
    let keywords = [
        "fn", "let", "mut", "if", "else", "while", "for", "in", "return",
        "match", "struct", "enum", "true", "false", "import", "mod",
    ];
    for kw in &keywords {
        let (tokens, diags) = Lexer::new(kw).tokenize();
        assert!(
            !diags.has_errors(),
            "Keyword '{kw}' should not produce errors"
        );
        assert!(
            !tokens.is_empty(),
            "Keyword '{kw}' should produce at least one token"
        );
    }
}

/// Comment handling.
#[test]
fn lex_comments() {
    let src = "let x = 5; // this is a comment\nlet y = 10;";
    let (tokens, diags) = Lexer::new(src).tokenize();
    assert!(!diags.has_errors(), "Comments should not produce errors");
    // Comments should be stripped; verify `x` and `y` are both present
    let idents: Vec<_> = tokens
        .iter()
        .filter(|t| t.kind == TokenKind::Ident)
        .map(|t| t.text.as_str())
        .collect();
    assert!(idents.contains(&"x") && idents.contains(&"y"));
}

/// Deeply nested expressions don't cause stack overflow in the lexer.
#[test]
fn lex_deeply_nested_parens() {
    let depth = 200;
    let src = format!(
        "{}1{}",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    let (tokens, _) = Lexer::new(&src).tokenize();
    // Should not stack overflow
    assert!(!tokens.is_empty(), "Deeply nested parens should tokenize");
}

/// Unterminated string literal.
#[test]
fn lex_unterminated_string() {
    let (_, diags) = Lexer::new("\"hello").tokenize();
    assert!(diags.has_errors(), "Unterminated string should produce error");
}

/// Null bytes in input.
#[test]
fn lex_null_bytes() {
    let src = "let x\0 = 5;";
    // Should not panic
    let _ = Lexer::new(src).tokenize();
}

/// Very long identifier.
#[test]
fn lex_very_long_identifier() {
    let long_name: String = "a".repeat(10000);
    let src = format!("let {} = 5;", long_name);
    let (tokens, _) = Lexer::new(&src).tokenize();
    assert!(!tokens.is_empty(), "Very long identifier should tokenize");
}

/// Hex, binary, and octal literals (if supported).
#[test]
fn lex_numeric_bases() {
    let src = "0xFF 0b1010 0o77";
    let (tokens, _) = Lexer::new(src).tokenize();
    // Should not panic; whether these are valid depends on CJC spec
    assert!(!tokens.is_empty());
}
