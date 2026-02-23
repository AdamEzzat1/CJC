// CJC Test Suite — cjc-lexer (14 tests)
// Source: crates/cjc-lexer/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_lexer::*;
use cjc_diag::Span;

fn lex(source: &str) -> Vec<Token> {
    let lexer = Lexer::new(source);
    let (tokens, _) = lexer.tokenize();
    tokens
}

fn kinds(source: &str) -> Vec<TokenKind> {
    lex(source).into_iter().map(|t| t.kind).collect()
}

#[test]
fn test_empty() {
    assert_eq!(kinds(""), vec![TokenKind::Eof]);
}

#[test]
fn test_keywords() {
    assert_eq!(
        kinds("struct class fn trait impl let mut return if else while nogc"),
        vec![
            TokenKind::Struct,
            TokenKind::Class,
            TokenKind::Fn,
            TokenKind::Trait,
            TokenKind::Impl,
            TokenKind::Let,
            TokenKind::Mut,
            TokenKind::Return,
            TokenKind::If,
            TokenKind::Else,
            TokenKind::While,
            TokenKind::NoGc,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_identifiers() {
    let tokens = lex("foo bar_baz _x T123");
    assert_eq!(tokens[0].kind, TokenKind::Ident);
    assert_eq!(tokens[0].text, "foo");
    assert_eq!(tokens[1].text, "bar_baz");
    assert_eq!(tokens[2].text, "_x");
    assert_eq!(tokens[3].text, "T123");
}

#[test]
fn test_numbers() {
    let tokens = lex("42 3.14 1_000 2.5e10 1f32 100i64");
    assert_eq!(tokens[0].kind, TokenKind::IntLit);
    assert_eq!(tokens[0].text, "42");
    assert_eq!(tokens[1].kind, TokenKind::FloatLit);
    assert_eq!(tokens[1].text, "3.14");
    assert_eq!(tokens[2].kind, TokenKind::IntLit);
    assert_eq!(tokens[2].text, "1_000");
    assert_eq!(tokens[3].kind, TokenKind::FloatLit);
    assert_eq!(tokens[4].kind, TokenKind::FloatLit);
    assert_eq!(tokens[5].kind, TokenKind::IntLit);
}

#[test]
fn test_strings() {
    let tokens = lex(r#""hello" "world\n" "tab\there""#);
    assert_eq!(tokens[0].kind, TokenKind::StringLit);
    assert_eq!(tokens[0].text, "hello");
    assert_eq!(tokens[1].text, "world\n");
    assert_eq!(tokens[2].text, "tab\there");
}

#[test]
fn test_operators() {
    assert_eq!(
        kinds("+ - * / % == != < > <= >= && || ! = |> -> =>"),
        vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::EqEq,
            TokenKind::BangEq,
            TokenKind::Lt,
            TokenKind::Gt,
            TokenKind::LtEq,
            TokenKind::GtEq,
            TokenKind::AmpAmp,
            TokenKind::PipePipe,
            TokenKind::Bang,
            TokenKind::Eq,
            TokenKind::PipeGt,
            TokenKind::Arrow,
            TokenKind::FatArrow,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_delimiters() {
    assert_eq!(
        kinds("( ) { } [ ] , . : ;"),
        vec![
            TokenKind::LParen,
            TokenKind::RParen,
            TokenKind::LBrace,
            TokenKind::RBrace,
            TokenKind::LBracket,
            TokenKind::RBracket,
            TokenKind::Comma,
            TokenKind::Dot,
            TokenKind::Colon,
            TokenKind::Semicolon,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_comments() {
    let tokens = lex("foo // this is a comment\nbar");
    assert_eq!(tokens[0].text, "foo");
    assert_eq!(tokens[1].text, "bar");
}

#[test]
fn test_block_comment() {
    let tokens = lex("foo /* block */ bar");
    assert_eq!(tokens[0].text, "foo");
    assert_eq!(tokens[1].text, "bar");
}

#[test]
fn test_nested_block_comment() {
    let tokens = lex("foo /* outer /* inner */ still comment */ bar");
    assert_eq!(tokens[0].text, "foo");
    assert_eq!(tokens[1].text, "bar");
}

#[test]
fn test_spans() {
    let tokens = lex("let x = 42;");
    assert_eq!(tokens[0].span, Span::new(0, 3)); // let
    assert_eq!(tokens[1].span, Span::new(4, 5)); // x
    assert_eq!(tokens[2].span, Span::new(6, 7)); // =
    assert_eq!(tokens[3].span, Span::new(8, 10)); // 42
    assert_eq!(tokens[4].span, Span::new(10, 11)); // ;
}

#[test]
fn test_unterminated_string() {
    let lexer = Lexer::new("\"hello");
    let (_, diags) = lexer.tokenize();
    assert!(diags.has_errors());
}

#[test]
fn test_function_signature() {
    assert_eq!(
        kinds("fn matmul<T: Float>(a: Tensor<T>, b: Tensor<T>) -> Tensor<T>"),
        vec![
            TokenKind::Fn,
            TokenKind::Ident,    // matmul
            TokenKind::Lt,       // <
            TokenKind::Ident,    // T
            TokenKind::Colon,    // :
            TokenKind::Ident,    // Float
            TokenKind::Gt,       // >
            TokenKind::LParen,   // (
            TokenKind::Ident,    // a
            TokenKind::Colon,    // :
            TokenKind::Ident,    // Tensor
            TokenKind::Lt,       // <
            TokenKind::Ident,    // T
            TokenKind::Gt,       // >
            TokenKind::Comma,    // ,
            TokenKind::Ident,    // b
            TokenKind::Colon,    // :
            TokenKind::Ident,    // Tensor
            TokenKind::Lt,       // <
            TokenKind::Ident,    // T
            TokenKind::Gt,       // >
            TokenKind::RParen,   // )
            TokenKind::Arrow,    // ->
            TokenKind::Ident,    // Tensor
            TokenKind::Lt,       // <
            TokenKind::Ident,    // T
            TokenKind::Gt,       // >
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_pipe_operator() {
    assert_eq!(
        kinds("df |> filter(x) |> group_by(y)"),
        vec![
            TokenKind::Ident,  // df
            TokenKind::PipeGt, // |>
            TokenKind::Ident,  // filter
            TokenKind::LParen,
            TokenKind::Ident,  // x
            TokenKind::RParen,
            TokenKind::PipeGt, // |>
            TokenKind::Ident,  // group_by
            TokenKind::LParen,
            TokenKind::Ident,  // y
            TokenKind::RParen,
            TokenKind::Eof,
        ]
    );
}
