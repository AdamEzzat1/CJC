//! Lexer-driven ANSI syntax highlighter for the CJC REPL.
//!
//! Reuses the CJC lexer to tokenize input text and wraps each token
//! in ANSI escape codes based on its `TokenKind`. Zero external dependencies.

use cjc_lexer::{Lexer, TokenKind};

// ── ANSI color constants ─────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const YELLOW: &str = "\x1b[33m";       // keywords
const BOLD_YELLOW: &str = "\x1b[1;33m"; // control flow keywords
const GREEN: &str = "\x1b[32m";        // strings
const CYAN: &str = "\x1b[36m";         // numbers
const BLUE: &str = "\x1b[34m";         // operators / punctuation
const MAGENTA: &str = "\x1b[35m";      // types / struct / enum / trait
const RED: &str = "\x1b[31m";          // errors
const BOLD_WHITE: &str = "\x1b[1;37m"; // boolean literals
const _DIM: &str = "\x1b[2m";          // reserved for comment highlighting

/// Returns the ANSI color code for a given token kind.
fn color_for(kind: TokenKind) -> &'static str {
    match kind {
        // Control flow — bold yellow
        TokenKind::If
        | TokenKind::Else
        | TokenKind::While
        | TokenKind::For
        | TokenKind::In
        | TokenKind::Match
        | TokenKind::Return
        | TokenKind::Break
        | TokenKind::Continue => BOLD_YELLOW,

        // Declaration keywords — yellow
        TokenKind::Fn
        | TokenKind::Let
        | TokenKind::Mut
        | TokenKind::Const
        | TokenKind::Import
        | TokenKind::Mod
        | TokenKind::As
        | TokenKind::Pub
        | TokenKind::Sealed
        | TokenKind::NoGc
        | TokenKind::Col => YELLOW,

        // Type-defining keywords — magenta
        TokenKind::Struct
        | TokenKind::Class
        | TokenKind::Record
        | TokenKind::Trait
        | TokenKind::Impl
        | TokenKind::Enum => MAGENTA,

        // Boolean / null literals — bold white
        TokenKind::True | TokenKind::False | TokenKind::Null => BOLD_WHITE,

        // Numeric literals — cyan
        TokenKind::IntLit | TokenKind::FloatLit => CYAN,

        // String literals — green
        TokenKind::StringLit
        | TokenKind::ByteStringLit
        | TokenKind::ByteCharLit
        | TokenKind::RawStringLit
        | TokenKind::RawByteStringLit
        | TokenKind::FStringLit
        | TokenKind::RegexLit => GREEN,

        // Operators — blue
        TokenKind::Plus
        | TokenKind::Minus
        | TokenKind::Star
        | TokenKind::Slash
        | TokenKind::Percent
        | TokenKind::StarStar
        | TokenKind::EqEq
        | TokenKind::BangEq
        | TokenKind::Lt
        | TokenKind::Gt
        | TokenKind::LtEq
        | TokenKind::GtEq
        | TokenKind::AmpAmp
        | TokenKind::PipePipe
        | TokenKind::Bang
        | TokenKind::Pipe
        | TokenKind::PipeGt
        | TokenKind::Amp
        | TokenKind::Caret
        | TokenKind::LtLt
        | TokenKind::GtGt
        | TokenKind::Tilde
        | TokenKind::TildeEq
        | TokenKind::BangTilde
        | TokenKind::Arrow
        | TokenKind::FatArrow
        | TokenKind::DotDot
        | TokenKind::DotDotDot => BLUE,

        // Assignment — blue (dimmer distinction possible later)
        TokenKind::Eq
        | TokenKind::PlusEq
        | TokenKind::MinusEq
        | TokenKind::StarEq
        | TokenKind::SlashEq
        | TokenKind::PercentEq
        | TokenKind::StarStarEq
        | TokenKind::AmpEq
        | TokenKind::PipeEq
        | TokenKind::CaretEq
        | TokenKind::LtLtEq
        | TokenKind::GtGtEq => BLUE,

        // Errors — red
        TokenKind::Error => RED,

        // Everything else (idents, delimiters, punctuation, EOF) — no color
        _ => "",
    }
}

/// Highlight a source string using the CJC lexer, returning an ANSI-colored string.
///
/// This function is designed for single-line REPL input but handles multi-line
/// text correctly. Gaps between tokens (whitespace) are preserved as-is.
pub fn highlight(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    let lexer = Lexer::new(input);
    let (tokens, _diags) = lexer.tokenize();

    let mut out = String::with_capacity(input.len() + 64);
    let mut last_end = 0usize;

    for tok in &tokens {
        if tok.kind == TokenKind::Eof {
            break;
        }

        let start = tok.span.start;
        let end = tok.span.end;

        // Preserve any gap (whitespace / characters between tokens)
        if start > last_end && start <= input.len() {
            out.push_str(&input[last_end..start]);
        }

        let color = color_for(tok.kind);
        let text = if end <= input.len() {
            &input[start..end]
        } else {
            &tok.text
        };

        if color.is_empty() {
            out.push_str(text);
        } else {
            out.push_str(color);
            out.push_str(text);
            out.push_str(RESET);
        }

        last_end = end;
    }

    // Trailing text after last token
    if last_end < input.len() {
        out.push_str(&input[last_end..]);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_empty() {
        assert_eq!(highlight(""), "");
    }

    #[test]
    fn test_highlight_plain_ident() {
        // Identifiers get no color wrapping
        let result = highlight("foo");
        assert!(!result.contains("\x1b["));
        assert!(result.contains("foo"));
    }

    #[test]
    fn test_highlight_keyword() {
        let result = highlight("let x = 42");
        // "let" should be yellow
        assert!(result.contains(&format!("{}let{}", YELLOW, RESET)));
        // "42" should be cyan
        assert!(result.contains(&format!("{}42{}", CYAN, RESET)));
    }

    #[test]
    fn test_highlight_string_literal() {
        let result = highlight("\"hello\"");
        assert!(result.contains(GREEN));
    }

    #[test]
    fn test_highlight_control_flow() {
        let result = highlight("if true { return 1 }");
        assert!(result.contains(&format!("{}if{}", BOLD_YELLOW, RESET)));
        assert!(result.contains(&format!("{}return{}", BOLD_YELLOW, RESET)));
        assert!(result.contains(&format!("{}true{}", BOLD_WHITE, RESET)));
    }

    #[test]
    fn test_highlight_operators() {
        let result = highlight("a + b == c");
        assert!(result.contains(&format!("{}+{}", BLUE, RESET)));
        assert!(result.contains(&format!("{}=={}", BLUE, RESET)));
    }

    #[test]
    fn test_highlight_struct_keyword() {
        let result = highlight("struct Foo");
        assert!(result.contains(&format!("{}struct{}", MAGENTA, RESET)));
    }

    #[test]
    fn test_highlight_preserves_whitespace() {
        let result = highlight("let   x");
        // The three spaces between "let" and "x" should be preserved
        assert!(result.contains("   "));
    }

    #[test]
    fn test_highlight_round_trip_content() {
        // Stripping all ANSI codes should yield the original text
        let input = "fn add(a: i64, b: i64) -> i64 { return a + b }";
        let highlighted = highlight(input);
        let stripped = strip_ansi(&highlighted);
        assert_eq!(stripped, input);
    }

    /// Helper: strip ANSI escape sequences from a string.
    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                // Skip until 'm'
                for inner in chars.by_ref() {
                    if inner == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }
}
