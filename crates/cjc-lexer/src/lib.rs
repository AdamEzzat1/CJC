use cjc_diag::{Diagnostic, DiagnosticBag, Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    // Literals
    IntLit,
    FloatLit,
    StringLit,
    ByteStringLit,   // b"..."
    ByteCharLit,     // b'c'
    RawStringLit,    // r"..." or r#"..."#
    RawByteStringLit,// br"..." or br#"..."#
    FStringLit,      // f"...{expr}..."  (string interpolation)
    RegexLit,        // /pattern/flags
    True,
    False,

    // Identifiers & keywords
    Ident,
    Struct,
    Class,
    Record,
    Fn,
    Trait,
    Impl,
    Let,
    Mut,
    Return,
    Break,
    Continue,
    If,
    Else,
    While,
    For,
    In,
    DotDot,    // ..
    DotDotDot, // ...
    NoGc,
    Col,
    Import,
    Mod,
    As,
    Sealed,
    Match,
    Enum,
    Const,
    Pub,
    Null,
    Underscore,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar,   // ** (power)
    EqEq,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    AmpAmp,
    PipePipe,
    Bang,
    Eq,
    Pipe,   // |
    PipeGt, // |>
    Question, // ?
    Tilde,     // ~
    TildeEq,   // ~=
    BangTilde, // !~
    // Compound assignment
    PlusEq,    // +=
    MinusEq,   // -=
    StarEq,    // *=
    SlashEq,   // /=
    PercentEq, // %=
    StarStarEq,// **=
    // Bitwise operators
    Amp,       // &  (bitwise AND)
    Caret,     // ^  (bitwise XOR)
    LtLt,      // << (left shift)
    GtGt,      // >> (right shift)
    // Bitwise compound assignment
    AmpEq,     // &=
    PipeEq,    // |=
    CaretEq,   // ^=
    LtLtEq,    // <<=
    GtGtEq,    // >>=

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LBracketPipe, // [|
    PipeRBracket, // |]

    // Punctuation
    Comma,
    Dot,
    Colon,
    Semicolon,
    Arrow, // ->
    FatArrow, // =>
    At,    // @

    // Special
    Eof,
    Error,
}

impl TokenKind {
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::Struct
                | TokenKind::Class
                | TokenKind::Fn
                | TokenKind::Trait
                | TokenKind::Impl
                | TokenKind::Let
                | TokenKind::Mut
                | TokenKind::Return
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::While
                | TokenKind::For
                | TokenKind::In
                | TokenKind::NoGc
                | TokenKind::Col
                | TokenKind::Import
                | TokenKind::Mod
                | TokenKind::As
                | TokenKind::Sealed
                | TokenKind::Match
                | TokenKind::Enum
                | TokenKind::Pub
                | TokenKind::True
                | TokenKind::False
        )
    }

    pub fn describe(&self) -> &'static str {
        match self {
            TokenKind::IntLit => "integer literal",
            TokenKind::FloatLit => "float literal",
            TokenKind::StringLit => "string literal",
            TokenKind::ByteStringLit => "byte string literal",
            TokenKind::ByteCharLit => "byte char literal",
            TokenKind::RawStringLit => "raw string literal",
            TokenKind::RawByteStringLit => "raw byte string literal",
            TokenKind::FStringLit => "format string literal",
            TokenKind::RegexLit => "regex literal",
            TokenKind::True => "`true`",
            TokenKind::False => "`false`",
            TokenKind::Ident => "identifier",
            TokenKind::Struct => "`struct`",
            TokenKind::Class => "`class`",
            TokenKind::Record => "`record`",
            TokenKind::Fn => "`fn`",
            TokenKind::Trait => "`trait`",
            TokenKind::Impl => "`impl`",
            TokenKind::Let => "`let`",
            TokenKind::Mut => "`mut`",
            TokenKind::Return => "`return`",
            TokenKind::Break => "`break`",
            TokenKind::Continue => "`continue`",
            TokenKind::If => "`if`",
            TokenKind::Else => "`else`",
            TokenKind::While => "`while`",
            TokenKind::For => "`for`",
            TokenKind::In => "`in`",
            TokenKind::DotDot => "`..`",
            TokenKind::DotDotDot => "`...`",
            TokenKind::NoGc => "`nogc`",
            TokenKind::Col => "`col`",
            TokenKind::Import => "`import`",
            TokenKind::Mod => "`mod`",
            TokenKind::As => "`as`",
            TokenKind::Sealed => "`sealed`",
            TokenKind::Match => "`match`",
            TokenKind::Enum => "`enum`",
            TokenKind::Const => "`const`",
            TokenKind::Pub => "`pub`",
            TokenKind::Null => "`null`",
            TokenKind::Underscore => "`_`",
            TokenKind::Plus => "`+`",
            TokenKind::Minus => "`-`",
            TokenKind::Star => "`*`",
            TokenKind::Slash => "`/`",
            TokenKind::Percent => "`%`",
            TokenKind::StarStar => "`**`",
            TokenKind::EqEq => "`==`",
            TokenKind::BangEq => "`!=`",
            TokenKind::Lt => "`<`",
            TokenKind::Gt => "`>`",
            TokenKind::LtEq => "`<=`",
            TokenKind::GtEq => "`>=`",
            TokenKind::AmpAmp => "`&&`",
            TokenKind::PipePipe => "`||`",
            TokenKind::Bang => "`!`",
            TokenKind::Eq => "`=`",
            TokenKind::Pipe => "`|`",
            TokenKind::PipeGt => "`|>`",
            TokenKind::Question => "`?`",
            TokenKind::Tilde => "`~`",
            TokenKind::TildeEq => "`~=`",
            TokenKind::BangTilde => "`!~`",
            // Compound assignment
            TokenKind::PlusEq => "`+=`",
            TokenKind::MinusEq => "`-=`",
            TokenKind::StarEq => "`*=`",
            TokenKind::SlashEq => "`/=`",
            TokenKind::PercentEq => "`%=`",
            TokenKind::StarStarEq => "`**=`",
            // Bitwise
            TokenKind::Amp => "`&`",
            TokenKind::Caret => "`^`",
            TokenKind::LtLt => "`<<`",
            TokenKind::GtGt => "`>>`",
            TokenKind::AmpEq => "`&=`",
            TokenKind::PipeEq => "`|=`",
            TokenKind::CaretEq => "`^=`",
            TokenKind::LtLtEq => "`<<=`",
            TokenKind::GtGtEq => "`>>=`",
            TokenKind::LParen => "`(`",
            TokenKind::RParen => "`)`",
            TokenKind::LBrace => "`{`",
            TokenKind::RBrace => "`}`",
            TokenKind::LBracket => "`[`",
            TokenKind::RBracket => "`]`",
            TokenKind::LBracketPipe => "`[|`",
            TokenKind::PipeRBracket => "`|]`",
            TokenKind::Comma => "`,`",
            TokenKind::Dot => "`.`",
            TokenKind::Colon => "`:`",
            TokenKind::Semicolon => "`;`",
            TokenKind::Arrow => "`->`",
            TokenKind::FatArrow => "`=>`",
            TokenKind::At => "`@`",
            TokenKind::Eof => "end of file",
            TokenKind::Error => "error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span, text: impl Into<String>) -> Self {
        Self {
            kind,
            span,
            text: text.into(),
        }
    }

    pub fn int_value(&self) -> i64 {
        let clean = self.text.replace('_', "");
        if clean.starts_with("0x") || clean.starts_with("0X") {
            i64::from_str_radix(&clean[2..], 16).unwrap_or(0)
        } else if clean.starts_with("0b") || clean.starts_with("0B") {
            i64::from_str_radix(&clean[2..], 2).unwrap_or(0)
        } else if clean.starts_with("0o") || clean.starts_with("0O") {
            i64::from_str_radix(&clean[2..], 8).unwrap_or(0)
        } else {
            clean.parse().unwrap_or(0)
        }
    }

    pub fn float_value(&self) -> f64 {
        self.text.replace('_', "").parse().unwrap_or(0.0)
    }
}

pub struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    pos: usize,
    prev_kind: TokenKind,
    pub diagnostics: DiagnosticBag,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            bytes: source.as_bytes(),
            pos: 0,
            prev_kind: TokenKind::Eof,
            diagnostics: DiagnosticBag::new(),
        }
    }

    pub fn tokenize(mut self) -> (Vec<Token>, DiagnosticBag) {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            let is_eof = tok.kind == TokenKind::Eof;
            self.prev_kind = tok.kind;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        (tokens, self.diagnostics)
    }

    /// Returns true if the previous token could produce a value (meaning `/`
    /// should be division, not a regex start).
    fn prev_is_value(&self) -> bool {
        matches!(
            self.prev_kind,
            TokenKind::IntLit
                | TokenKind::FloatLit
                | TokenKind::StringLit
                | TokenKind::ByteStringLit
                | TokenKind::ByteCharLit
                | TokenKind::RawStringLit
                | TokenKind::RawByteStringLit
                | TokenKind::FStringLit
                | TokenKind::RegexLit
                | TokenKind::True
                | TokenKind::False
                | TokenKind::Ident
                | TokenKind::RParen
                | TokenKind::RBracket
                | TokenKind::PipeRBracket
                | TokenKind::RBrace
        )
    }

    fn peek(&self) -> u8 {
        if self.pos < self.bytes.len() {
            self.bytes[self.pos]
        } else {
            0
        }
    }

    fn peek_at(&self, offset: usize) -> u8 {
        let idx = self.pos + offset;
        if idx < self.bytes.len() {
            self.bytes[idx]
        } else {
            0
        }
    }

    fn advance(&mut self) -> u8 {
        let ch = self.peek();
        self.pos += 1;
        ch
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while self.pos < self.bytes.len() && self.peek().is_ascii_whitespace() {
                self.advance();
            }

            // Skip line comments
            if self.pos + 1 < self.bytes.len()
                && self.bytes[self.pos] == b'/'
                && self.bytes[self.pos + 1] == b'/'
            {
                while self.pos < self.bytes.len() && self.peek() != b'\n' {
                    self.advance();
                }
                continue;
            }

            // Skip block comments
            if self.pos + 1 < self.bytes.len()
                && self.bytes[self.pos] == b'/'
                && self.bytes[self.pos + 1] == b'*'
            {
                self.advance(); // /
                self.advance(); // *
                let mut depth = 1;
                while self.pos < self.bytes.len() && depth > 0 {
                    if self.peek() == b'/' && self.peek_at(1) == b'*' {
                        depth += 1;
                        self.advance();
                        self.advance();
                    } else if self.peek() == b'*' && self.peek_at(1) == b'/' {
                        depth -= 1;
                        self.advance();
                        self.advance();
                    } else {
                        self.advance();
                    }
                }
                continue;
            }

            break;
        }
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        let start = self.pos;

        if self.pos >= self.bytes.len() {
            return Token::new(TokenKind::Eof, Span::new(start, start), "");
        }

        let ch = self.advance();

        match ch {
            // Single-character tokens
            b'(' => Token::new(TokenKind::LParen, Span::new(start, self.pos), "("),
            b')' => Token::new(TokenKind::RParen, Span::new(start, self.pos), ")"),
            b'{' => Token::new(TokenKind::LBrace, Span::new(start, self.pos), "{"),
            b'}' => Token::new(TokenKind::RBrace, Span::new(start, self.pos), "}"),
            b'[' => {
                if self.pos < self.bytes.len() && self.bytes[self.pos] == b'|' {
                    self.pos += 1;
                    Token::new(TokenKind::LBracketPipe, Span::new(start, self.pos), "[|")
                } else {
                    Token::new(TokenKind::LBracket, Span::new(start, self.pos), "[")
                }
            }
            b']' => Token::new(TokenKind::RBracket, Span::new(start, self.pos), "]"),
            b',' => Token::new(TokenKind::Comma, Span::new(start, self.pos), ","),
            b'.' => {
                if self.peek() == b'.' {
                    self.advance();
                    if self.peek() == b'.' {
                        self.advance();
                        Token::new(TokenKind::DotDotDot, Span::new(start, self.pos), "...")
                    } else {
                        Token::new(TokenKind::DotDot, Span::new(start, self.pos), "..")
                    }
                } else {
                    Token::new(TokenKind::Dot, Span::new(start, self.pos), ".")
                }
            }
            b':' => Token::new(TokenKind::Colon, Span::new(start, self.pos), ":"),
            b';' => Token::new(TokenKind::Semicolon, Span::new(start, self.pos), ";"),
            b'+' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::PlusEq, Span::new(start, self.pos), "+=")
                } else {
                    Token::new(TokenKind::Plus, Span::new(start, self.pos), "+")
                }
            }
            b'*' => {
                if self.peek() == b'*' {
                    self.advance();
                    if self.peek() == b'=' {
                        self.advance();
                        Token::new(TokenKind::StarStarEq, Span::new(start, self.pos), "**=")
                    } else {
                        Token::new(TokenKind::StarStar, Span::new(start, self.pos), "**")
                    }
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::StarEq, Span::new(start, self.pos), "*=")
                } else {
                    Token::new(TokenKind::Star, Span::new(start, self.pos), "*")
                }
            }
            b'%' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::PercentEq, Span::new(start, self.pos), "%=")
                } else {
                    Token::new(TokenKind::Percent, Span::new(start, self.pos), "%")
                }
            }
            b'^' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::CaretEq, Span::new(start, self.pos), "^=")
                } else {
                    Token::new(TokenKind::Caret, Span::new(start, self.pos), "^")
                }
            }
            b'/' => {
                if !self.prev_is_value() && self.pos < self.bytes.len() && self.peek() != b'/' && self.peek() != b'*' && self.peek() != b' ' && self.peek() != b'\t' && self.peek() != b'\n' && self.peek() != b'\r' {
                    // Try regex lex with backtracking
                    let save_pos = self.pos;
                    let save_diag_count = self.diagnostics.count();
                    let tok = self.lex_regex(start);
                    if tok.kind == TokenKind::Error {
                        // Backtrack: not a regex, it's division
                        self.pos = save_pos;
                        self.diagnostics.truncate(save_diag_count);
                        Token::new(TokenKind::Slash, Span::new(start, save_pos), "/")
                    } else {
                        tok
                    }
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::SlashEq, Span::new(start, self.pos), "/=")
                } else {
                    Token::new(TokenKind::Slash, Span::new(start, self.pos), "/")
                }
            }
            b'?' => Token::new(TokenKind::Question, Span::new(start, self.pos), "?"),
            b'@' => Token::new(TokenKind::At, Span::new(start, self.pos), "@"),

            // Multi-character operators
            b'-' => {
                if self.peek() == b'>' {
                    self.advance();
                    Token::new(TokenKind::Arrow, Span::new(start, self.pos), "->")
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::MinusEq, Span::new(start, self.pos), "-=")
                } else {
                    Token::new(TokenKind::Minus, Span::new(start, self.pos), "-")
                }
            }
            b'=' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::EqEq, Span::new(start, self.pos), "==")
                } else if self.peek() == b'>' {
                    self.advance();
                    Token::new(TokenKind::FatArrow, Span::new(start, self.pos), "=>")
                } else {
                    Token::new(TokenKind::Eq, Span::new(start, self.pos), "=")
                }
            }
            b'!' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::BangEq, Span::new(start, self.pos), "!=")
                } else if self.peek() == b'~' {
                    self.advance();
                    Token::new(TokenKind::BangTilde, Span::new(start, self.pos), "!~")
                } else {
                    Token::new(TokenKind::Bang, Span::new(start, self.pos), "!")
                }
            }
            b'<' => {
                if self.peek() == b'<' {
                    self.advance();
                    if self.peek() == b'=' {
                        self.advance();
                        Token::new(TokenKind::LtLtEq, Span::new(start, self.pos), "<<=")
                    } else {
                        Token::new(TokenKind::LtLt, Span::new(start, self.pos), "<<")
                    }
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::LtEq, Span::new(start, self.pos), "<=")
                } else {
                    Token::new(TokenKind::Lt, Span::new(start, self.pos), "<")
                }
            }
            b'>' => {
                if self.peek() == b'>' {
                    self.advance();
                    if self.peek() == b'=' {
                        self.advance();
                        Token::new(TokenKind::GtGtEq, Span::new(start, self.pos), ">>=")
                    } else {
                        Token::new(TokenKind::GtGt, Span::new(start, self.pos), ">>")
                    }
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::GtEq, Span::new(start, self.pos), ">=")
                } else {
                    Token::new(TokenKind::Gt, Span::new(start, self.pos), ">")
                }
            }
            b'&' => {
                if self.peek() == b'&' {
                    self.advance();
                    Token::new(TokenKind::AmpAmp, Span::new(start, self.pos), "&&")
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::AmpEq, Span::new(start, self.pos), "&=")
                } else {
                    Token::new(TokenKind::Amp, Span::new(start, self.pos), "&")
                }
            }
            b'|' => {
                if self.peek() == b'|' {
                    self.advance();
                    Token::new(TokenKind::PipePipe, Span::new(start, self.pos), "||")
                } else if self.peek() == b'>' {
                    self.advance();
                    Token::new(TokenKind::PipeGt, Span::new(start, self.pos), "|>")
                } else if self.peek() == b']' {
                    self.advance();
                    Token::new(TokenKind::PipeRBracket, Span::new(start, self.pos), "|]")
                } else if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::PipeEq, Span::new(start, self.pos), "|=")
                } else {
                    // Single pipe — used for lambda parameter delimiters and bitwise OR
                    Token::new(TokenKind::Pipe, Span::new(start, self.pos), "|")
                }
            }

            // String literals
            b'"' => self.lex_string(start),

            // Byte string/char literals: b"...", b'c', br"..."
            b'b' if self.peek() == b'"' => {
                self.advance(); // consume opening "
                self.lex_byte_string(start)
            }
            b'b' if self.peek() == b'\'' => {
                self.advance(); // consume opening '
                self.lex_byte_char(start)
            }
            b'b' if self.peek() == b'r' && self.peek_at(1) == b'"' => {
                self.advance(); // consume r
                self.advance(); // consume "
                self.lex_raw_string(start, true, 0)
            }
            b'b' if self.peek() == b'r' && self.peek_at(1) == b'#' => {
                self.advance(); // consume r
                // Count # delimiters
                let mut hashes = 0;
                while self.peek() == b'#' {
                    self.advance();
                    hashes += 1;
                }
                if self.peek() == b'"' {
                    self.advance(); // consume opening "
                    self.lex_raw_string(start, true, hashes)
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0010",
                        "expected `\"` after `br` and `#` delimiters",
                        Span::new(start, self.pos),
                    ));
                    Token::new(TokenKind::Error, Span::new(start, self.pos), &self.source[start..self.pos])
                }
            }

            // Raw string literals: r"...", r#"..."#
            b'r' if self.peek() == b'"' => {
                self.advance(); // consume opening "
                self.lex_raw_string(start, false, 0)
            }
            b'r' if self.peek() == b'#' => {
                let save_pos = self.pos;
                let mut hashes = 0;
                while self.peek() == b'#' {
                    self.advance();
                    hashes += 1;
                }
                if self.peek() == b'"' {
                    self.advance(); // consume opening "
                    self.lex_raw_string(start, false, hashes)
                } else {
                    // Not a raw string — backtrack and lex as identifier
                    self.pos = save_pos;
                    self.lex_ident(start)
                }
            }

            // Format string literals: f"...{expr}..."
            b'f' if self.peek() == b'"' => {
                self.advance(); // consume opening "
                self.lex_fstring(start)
            }

            // Tilde operators
            b'~' => {
                if self.peek() == b'=' {
                    self.advance();
                    Token::new(TokenKind::TildeEq, Span::new(start, self.pos), "~=")
                } else {
                    Token::new(TokenKind::Tilde, Span::new(start, self.pos), "~")
                }
            }

            // Number literals
            b'0'..=b'9' => self.lex_number(start),

            // Identifiers and keywords
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_ident(start),

            _ => {
                self.diagnostics.emit(Diagnostic::error(
                    "E0002",
                    format!("unexpected character `{}`", ch as char),
                    Span::new(start, self.pos),
                ));
                Token::new(
                    TokenKind::Error,
                    Span::new(start, self.pos),
                    &self.source[start..self.pos],
                )
            }
        }
    }

    fn lex_string(&mut self, start: usize) -> Token {
        let mut value = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                self.diagnostics.emit(
                    Diagnostic::error(
                        "E0003",
                        "unterminated string literal",
                        Span::new(start, self.pos),
                    )
                    .with_hint("add a closing `\"` to terminate the string"),
                );
                break;
            }
            let ch = self.advance();
            match ch {
                b'"' => break,
                b'\\' => {
                    if self.pos >= self.bytes.len() {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0003",
                            "unterminated escape sequence",
                            Span::new(self.pos - 1, self.pos),
                        ));
                        break;
                    }
                    let esc = self.advance();
                    match esc {
                        b'n' => value.push('\n'),
                        b't' => value.push('\t'),
                        b'r' => value.push('\r'),
                        b'\\' => value.push('\\'),
                        b'"' => value.push('"'),
                        b'0' => value.push('\0'),
                        _ => {
                            self.diagnostics.emit(
                                Diagnostic::error(
                                    "E0004",
                                    format!("unknown escape sequence `\\{}`", esc as char),
                                    Span::new(self.pos - 2, self.pos),
                                )
                                .with_hint("valid escapes: \\n, \\t, \\r, \\\\, \\\", \\0"),
                            );
                            value.push(esc as char);
                        }
                    }
                }
                _ => value.push(ch as char),
            }
        }
        Token::new(TokenKind::StringLit, Span::new(start, self.pos), value)
    }

    /// Lex a format string literal: f"...{expr}..." (opening f" already consumed).
    /// The token text is the raw content between the outer quotes — the parser
    /// will split it on `{...}` interpolation holes and build a concat expr.
    fn lex_fstring(&mut self, start: usize) -> Token {
        let mut raw = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                self.diagnostics.emit(
                    Diagnostic::error(
                        "E0003",
                        "unterminated format string literal",
                        Span::new(start, self.pos),
                    )
                    .with_hint("add a closing `\"` to terminate the format string"),
                );
                break;
            }
            let ch = self.advance();
            match ch {
                b'"' => break,
                b'\\' => {
                    if self.pos >= self.bytes.len() {
                        break;
                    }
                    let esc = self.advance();
                    raw.push('\\');
                    raw.push(esc as char);
                }
                b'{' => {
                    // `{{` is an escaped literal `{`
                    if self.pos < self.bytes.len() && self.bytes[self.pos] == b'{' {
                        self.advance();
                        raw.push('{');
                    } else {
                        // Start of interpolation hole — collect verbatim until matching `}`
                        // We include the braces so the parser can see them.
                        raw.push('{');
                        let mut depth = 1usize;
                        while self.pos < self.bytes.len() && depth > 0 {
                            let inner = self.advance();
                            raw.push(inner as char);
                            if inner == b'{' {
                                depth += 1;
                            } else if inner == b'}' {
                                depth -= 1;
                            }
                        }
                    }
                }
                b'}' => {
                    // `}}` is an escaped literal `}`
                    if self.pos < self.bytes.len() && self.bytes[self.pos] == b'}' {
                        self.advance();
                        raw.push('}');
                    } else {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0450",
                            "unexpected `}` in format string (use `}}` for a literal `}`)",
                            Span::new(self.pos - 1, self.pos),
                        ));
                        raw.push('}');
                    }
                }
                _ => raw.push(ch as char),
            }
        }
        Token::new(TokenKind::FStringLit, Span::new(start, self.pos), raw)
    }

    /// Lex a byte string literal: b"..." (opening b" already consumed).
    fn lex_byte_string(&mut self, start: usize) -> Token {
        let mut value = Vec::new();
        loop {
            if self.pos >= self.bytes.len() {
                self.diagnostics.emit(
                    Diagnostic::error(
                        "E0003",
                        "unterminated byte string literal",
                        Span::new(start, self.pos),
                    )
                    .with_hint("add a closing `\"` to terminate the byte string"),
                );
                break;
            }
            let ch = self.advance();
            match ch {
                b'"' => break,
                b'\\' => {
                    if self.pos >= self.bytes.len() {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0003",
                            "unterminated escape sequence in byte string",
                            Span::new(self.pos - 1, self.pos),
                        ));
                        break;
                    }
                    let esc = self.advance();
                    match esc {
                        b'n' => value.push(b'\n'),
                        b't' => value.push(b'\t'),
                        b'r' => value.push(b'\r'),
                        b'\\' => value.push(b'\\'),
                        b'"' => value.push(b'"'),
                        b'0' => value.push(0),
                        b'x' => {
                            // \xNN hex byte
                            if self.pos + 1 < self.bytes.len() {
                                let hi = self.advance();
                                let lo = self.advance();
                                match (hex_digit(hi), hex_digit(lo)) {
                                    (Some(h), Some(l)) => value.push(h * 16 + l),
                                    _ => {
                                        self.diagnostics.emit(
                                            Diagnostic::error(
                                                "E0004",
                                                format!("invalid hex escape `\\x{}{}`", hi as char, lo as char),
                                                Span::new(self.pos - 4, self.pos),
                                            )
                                            .with_hint("hex escapes must be \\xNN where N is 0-9 or a-f"),
                                        );
                                    }
                                }
                            } else {
                                self.diagnostics.emit(Diagnostic::error(
                                    "E0004",
                                    "incomplete hex escape in byte string",
                                    Span::new(self.pos - 2, self.pos),
                                ));
                            }
                        }
                        _ => {
                            self.diagnostics.emit(
                                Diagnostic::error(
                                    "E0004",
                                    format!("unknown escape sequence `\\{}`", esc as char),
                                    Span::new(self.pos - 2, self.pos),
                                )
                                .with_hint("valid escapes: \\n, \\t, \\r, \\\\, \\\", \\0, \\xNN"),
                            );
                            value.push(esc);
                        }
                    }
                }
                _ => value.push(ch),
            }
        }
        // Store the byte values as a string for token text (bytes as latin-1 chars)
        let text: String = value.iter().map(|&b| b as char).collect();
        Token::new(TokenKind::ByteStringLit, Span::new(start, self.pos), text)
    }

    /// Lex a byte char literal: b'c' (opening b' already consumed).
    fn lex_byte_char(&mut self, start: usize) -> Token {
        if self.pos >= self.bytes.len() {
            self.diagnostics.emit(Diagnostic::error(
                "E0003",
                "unterminated byte char literal",
                Span::new(start, self.pos),
            ));
            return Token::new(TokenKind::Error, Span::new(start, self.pos), "");
        }

        let byte_val = if self.peek() == b'\\' {
            self.advance(); // consume backslash
            if self.pos >= self.bytes.len() {
                self.diagnostics.emit(Diagnostic::error(
                    "E0003",
                    "unterminated escape in byte char literal",
                    Span::new(start, self.pos),
                ));
                return Token::new(TokenKind::Error, Span::new(start, self.pos), "");
            }
            let esc = self.advance();
            match esc {
                b'n' => b'\n',
                b't' => b'\t',
                b'r' => b'\r',
                b'\\' => b'\\',
                b'\'' => b'\'',
                b'0' => 0,
                b'x' => {
                    if self.pos + 1 < self.bytes.len() {
                        let hi = self.advance();
                        let lo = self.advance();
                        match (hex_digit(hi), hex_digit(lo)) {
                            (Some(h), Some(l)) => h * 16 + l,
                            _ => {
                                self.diagnostics.emit(Diagnostic::error(
                                    "E0004",
                                    format!("invalid hex escape in byte char `\\x{}{}`", hi as char, lo as char),
                                    Span::new(self.pos - 4, self.pos),
                                ));
                                0
                            }
                        }
                    } else {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0004",
                            "incomplete hex escape in byte char",
                            Span::new(self.pos - 2, self.pos),
                        ));
                        0
                    }
                }
                _ => {
                    self.diagnostics.emit(
                        Diagnostic::error(
                            "E0004",
                            format!("unknown escape `\\{}` in byte char literal", esc as char),
                            Span::new(self.pos - 2, self.pos),
                        )
                        .with_hint("valid escapes: \\n, \\t, \\r, \\\\, \\', \\0, \\xNN"),
                    );
                    esc
                }
            }
        } else {
            self.advance()
        };

        // Expect closing '
        if self.pos < self.bytes.len() && self.peek() == b'\'' {
            self.advance();
        } else {
            self.diagnostics.emit(Diagnostic::error(
                "E0003",
                "unterminated byte char literal, expected closing `'`",
                Span::new(start, self.pos),
            ));
            return Token::new(TokenKind::Error, Span::new(start, self.pos), "");
        }

        // Store the byte value as text (decimal string)
        let text = byte_val.to_string();
        Token::new(TokenKind::ByteCharLit, Span::new(start, self.pos), text)
    }

    /// Lex a raw string: r"..." or r#"..."# (or br"..." / br#"..."#).
    /// `is_byte` determines if this is a raw byte string or raw string.
    /// `hashes` is the number of # delimiters (0 for r"...").
    /// The opening quote has already been consumed.
    fn lex_raw_string(&mut self, start: usize, is_byte: bool, hashes: usize) -> Token {
        let mut value = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                self.diagnostics.emit(
                    Diagnostic::error(
                        "E0003",
                        "unterminated raw string literal",
                        Span::new(start, self.pos),
                    )
                    .with_hint(if hashes > 0 {
                        format!("add a closing `\"{}`", "#".repeat(hashes))
                    } else {
                        "add a closing `\"` to terminate the raw string".to_string()
                    }),
                );
                break;
            }
            let ch = self.advance();
            if ch == b'"' {
                // Check if followed by the right number of #
                let mut found_hashes = 0;
                while found_hashes < hashes && self.pos < self.bytes.len() && self.peek() == b'#' {
                    self.advance();
                    found_hashes += 1;
                }
                if found_hashes == hashes {
                    break; // End of raw string
                }
                // Not enough # — the " and any # we consumed are part of the string
                value.push('"');
                for _ in 0..found_hashes {
                    value.push('#');
                }
            } else {
                value.push(ch as char);
            }
        }
        let kind = if is_byte {
            TokenKind::RawByteStringLit
        } else {
            TokenKind::RawStringLit
        };
        Token::new(kind, Span::new(start, self.pos), value)
    }

    /// Lex a regex literal: /pattern/flags
    /// The opening `/` has already been consumed.
    /// Flags: `i` (case-insensitive), `g` (global), `m` (multiline), `s` (dotall), `x` (extended).
    fn lex_regex(&mut self, start: usize) -> Token {
        let mut pattern = String::new();
        loop {
            if self.pos >= self.bytes.len() {
                self.diagnostics.emit(
                    Diagnostic::error(
                        "E0011",
                        "unterminated regex literal",
                        Span::new(start, self.pos),
                    )
                    .with_hint("add a closing `/` to terminate the regex"),
                );
                let text = format!("/{}", pattern);
                return Token::new(TokenKind::Error, Span::new(start, self.pos), text);
            }
            let ch = self.advance();
            match ch {
                b'/' => break,
                b'\\' => {
                    // Escape next char in regex (pass through to engine)
                    pattern.push('\\');
                    if self.pos < self.bytes.len() {
                        let esc = self.advance();
                        pattern.push(esc as char);
                    } else {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0011",
                            "unterminated escape in regex literal",
                            Span::new(self.pos - 1, self.pos),
                        ));
                        let text = format!("/{}", pattern);
                        return Token::new(TokenKind::Error, Span::new(start, self.pos), text);
                    }
                }
                b'\n' => {
                    // Newlines not allowed in regex literals (use `x` flag for multiline)
                    self.diagnostics.emit(
                        Diagnostic::error(
                            "E0011",
                            "newline in regex literal",
                            Span::new(self.pos - 1, self.pos),
                        )
                        .with_hint("use the `x` flag for extended mode with whitespace"),
                    );
                    let text = format!("/{}", pattern);
                    return Token::new(TokenKind::Error, Span::new(start, self.pos), text);
                }
                _ => pattern.push(ch as char),
            }
        }
        // Parse flags after closing /
        let mut flags = String::new();
        while self.pos < self.bytes.len() && matches!(self.peek(), b'i' | b'g' | b'm' | b's' | b'x') {
            flags.push(self.advance() as char);
        }
        // Token text format: "pattern" + "\0" + "flags" (flags may be empty)
        let text = if flags.is_empty() {
            pattern
        } else {
            format!("{}\0{}", pattern, flags)
        };
        Token::new(TokenKind::RegexLit, Span::new(start, self.pos), text)
    }

    fn lex_number(&mut self, start: usize) -> Token {
        let mut is_float = false;

        // Check for hex (0x), binary (0b), or octal (0o) prefixes
        // Note: at this point, `start` points to the first digit and we've already
        // advanced past it, so self.source[start..self.pos] is "0".
        if self.source[start..self.pos].starts_with('0') && self.pos < self.bytes.len() {
            match self.peek() {
                b'x' | b'X' => {
                    self.advance(); // consume 'x'
                    let digit_start = self.pos;
                    while self.pos < self.bytes.len()
                        && (self.peek().is_ascii_hexdigit() || self.peek() == b'_')
                    {
                        self.advance();
                    }
                    if self.pos == digit_start {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0006",
                            "expected hex digits after `0x`",
                            Span::new(start, self.pos),
                        ));
                        return Token::new(TokenKind::Error, Span::new(start, self.pos), &self.source[start..self.pos]);
                    }
                    let text = &self.source[start..self.pos];
                    return Token::new(TokenKind::IntLit, Span::new(start, self.pos), text);
                }
                b'b' | b'B' if self.peek_at(1) != b'"' && self.peek_at(1) != b'\'' && self.peek_at(1) != b'r' => {
                    self.advance(); // consume 'b'
                    let digit_start = self.pos;
                    while self.pos < self.bytes.len()
                        && (self.peek() == b'0' || self.peek() == b'1' || self.peek() == b'_')
                    {
                        self.advance();
                    }
                    if self.pos == digit_start {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0006",
                            "expected binary digits after `0b`",
                            Span::new(start, self.pos),
                        ));
                        return Token::new(TokenKind::Error, Span::new(start, self.pos), &self.source[start..self.pos]);
                    }
                    let text = &self.source[start..self.pos];
                    return Token::new(TokenKind::IntLit, Span::new(start, self.pos), text);
                }
                b'o' | b'O' => {
                    self.advance(); // consume 'o'
                    let digit_start = self.pos;
                    while self.pos < self.bytes.len()
                        && ((self.peek() >= b'0' && self.peek() <= b'7') || self.peek() == b'_')
                    {
                        self.advance();
                    }
                    if self.pos == digit_start {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0006",
                            "expected octal digits after `0o`",
                            Span::new(start, self.pos),
                        ));
                        return Token::new(TokenKind::Error, Span::new(start, self.pos), &self.source[start..self.pos]);
                    }
                    let text = &self.source[start..self.pos];
                    return Token::new(TokenKind::IntLit, Span::new(start, self.pos), text);
                }
                _ => {}
            }
        }

        // Integer part
        while self.pos < self.bytes.len()
            && (self.peek().is_ascii_digit() || self.peek() == b'_')
        {
            self.advance();
        }

        // Fractional part
        if self.peek() == b'.' && self.peek_at(1).is_ascii_digit() {
            is_float = true;
            self.advance(); // .
            while self.pos < self.bytes.len()
                && (self.peek().is_ascii_digit() || self.peek() == b'_')
            {
                self.advance();
            }
        }

        // Exponent
        if self.peek() == b'e' || self.peek() == b'E' {
            is_float = true;
            self.advance();
            if self.peek() == b'+' || self.peek() == b'-' {
                self.advance();
            }
            while self.pos < self.bytes.len() && self.peek().is_ascii_digit() {
                self.advance();
            }
        }

        // Type suffix: f32, f64, i32, i64
        if self.peek() == b'f' || self.peek() == b'i' {
            let suffix_start = self.pos;
            self.advance();
            while self.pos < self.bytes.len() && self.peek().is_ascii_digit() {
                self.advance();
            }
            let suffix = &self.source[suffix_start..self.pos];
            match suffix {
                "f32" | "f64" => is_float = true,
                "i32" | "i64" => {}
                _ => {
                    self.diagnostics.emit(Diagnostic::error(
                        "E0005",
                        format!("invalid numeric suffix `{}`", suffix),
                        Span::new(suffix_start, self.pos),
                    ));
                }
            }
        }

        let text = &self.source[start..self.pos];
        let kind = if is_float {
            TokenKind::FloatLit
        } else {
            TokenKind::IntLit
        };
        Token::new(kind, Span::new(start, self.pos), text)
    }

    fn lex_ident(&mut self, start: usize) -> Token {
        while self.pos < self.bytes.len()
            && (self.peek().is_ascii_alphanumeric() || self.peek() == b'_')
        {
            self.advance();
        }

        let text = &self.source[start..self.pos];
        let kind = match text {
            "struct" => TokenKind::Struct,
            "class" => TokenKind::Class,
            "record" => TokenKind::Record,
            "fn" => TokenKind::Fn,
            "trait" => TokenKind::Trait,
            "impl" => TokenKind::Impl,
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "nogc" => TokenKind::NoGc,
            "col" => TokenKind::Col,
            "import" => TokenKind::Import,
            "mod" => TokenKind::Mod,
            "as" => TokenKind::As,
            "sealed" => TokenKind::Sealed,
            "match" => TokenKind::Match,
            "enum" => TokenKind::Enum,
            "const" => TokenKind::Const,
            "pub" => TokenKind::Pub,
            "null" => TokenKind::Null,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "_" => TokenKind::Underscore,
            _ => TokenKind::Ident,
        };

        Token::new(kind, Span::new(start, self.pos), text)
    }
}

/// Convert a hex digit (ASCII) to its numeric value.
fn hex_digit(ch: u8) -> Option<u8> {
    match ch {
        b'0'..=b'9' => Some(ch - b'0'),
        b'a'..=b'f' => Some(ch - b'a' + 10),
        b'A'..=b'F' => Some(ch - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_pipe_token_for_lambda() {
        assert_eq!(
            kinds("|x: f64| x * 2.0"),
            vec![
                TokenKind::Pipe,     // |
                TokenKind::Ident,    // x
                TokenKind::Colon,    // :
                TokenKind::Ident,    // f64
                TokenKind::Pipe,     // |
                TokenKind::Ident,    // x
                TokenKind::Star,     // *
                TokenKind::FloatLit, // 2.0
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_pipe_no_error() {
        let lexer = Lexer::new("|x| x");
        let (tokens, diags) = lexer.tokenize();
        assert!(!diags.has_errors());
        assert_eq!(tokens[0].kind, TokenKind::Pipe);
        assert_eq!(tokens[2].kind, TokenKind::Pipe);
    }

    // ── Byte string literal tests ──────────────────────────────────

    #[test]
    fn test_byte_string_basic() {
        let tokens = lex(r#"b"hello""#);
        assert_eq!(tokens[0].kind, TokenKind::ByteStringLit);
        assert_eq!(tokens[0].text, "hello");
    }

    #[test]
    fn test_byte_string_escapes() {
        let tokens = lex(r#"b"a\nb\t""#);
        assert_eq!(tokens[0].kind, TokenKind::ByteStringLit);
        assert_eq!(tokens[0].text, "a\nb\t");
    }

    #[test]
    fn test_byte_string_hex_escape() {
        let tokens = lex(r#"b"\xff\x00\x41""#);
        assert_eq!(tokens[0].kind, TokenKind::ByteStringLit);
        // \xff = 255, \x00 = 0, \x41 = 65 ('A')
        let bytes: Vec<u8> = tokens[0].text.chars().map(|c| c as u8).collect();
        assert_eq!(bytes, vec![0xff, 0x00, 0x41]);
    }

    #[test]
    fn test_byte_string_span() {
        let tokens = lex(r#"b"abc""#);
        assert_eq!(tokens[0].span, Span::new(0, 6)); // b"abc" is 6 chars
    }

    #[test]
    fn test_byte_string_unterminated() {
        let lexer = Lexer::new(r#"b"hello"#);
        let (_, diags) = lexer.tokenize();
        assert!(diags.has_errors());
    }

    // ── Byte char literal tests ────────────────────────────────────

    #[test]
    fn test_byte_char_basic() {
        let tokens = lex("b'A'");
        assert_eq!(tokens[0].kind, TokenKind::ByteCharLit);
        assert_eq!(tokens[0].text, "65"); // 'A' = 65
    }

    #[test]
    fn test_byte_char_newline() {
        let tokens = lex(r"b'\n'");
        assert_eq!(tokens[0].kind, TokenKind::ByteCharLit);
        assert_eq!(tokens[0].text, "10"); // '\n' = 10
    }

    #[test]
    fn test_byte_char_null() {
        let tokens = lex(r"b'\0'");
        assert_eq!(tokens[0].kind, TokenKind::ByteCharLit);
        assert_eq!(tokens[0].text, "0");
    }

    #[test]
    fn test_byte_char_hex() {
        let tokens = lex(r"b'\xff'");
        assert_eq!(tokens[0].kind, TokenKind::ByteCharLit);
        assert_eq!(tokens[0].text, "255");
    }

    #[test]
    fn test_byte_char_backslash() {
        let tokens = lex(r"b'\\'");
        assert_eq!(tokens[0].kind, TokenKind::ByteCharLit);
        assert_eq!(tokens[0].text, "92"); // '\\' = 92
    }

    #[test]
    fn test_byte_char_unterminated() {
        let lexer = Lexer::new("b'A");
        let (_, diags) = lexer.tokenize();
        assert!(diags.has_errors());
    }

    // ── Raw string literal tests ───────────────────────────────────

    #[test]
    fn test_raw_string_basic() {
        let tokens = lex(r#"r"hello\nworld""#);
        assert_eq!(tokens[0].kind, TokenKind::RawStringLit);
        // Raw strings preserve backslashes literally
        assert_eq!(tokens[0].text, r"hello\nworld");
    }

    #[test]
    fn test_raw_string_with_hashes() {
        // r#"She said "hi""#
        let source = "r#\"She said \\\"hi\\\"\"#";
        // Actually, let's construct this more carefully for a real raw string
        // The source text is: r#"contains "quotes""#
        let source2 = r###"r#"contains "quotes""#"###;
        let tokens = lex(source2);
        assert_eq!(tokens[0].kind, TokenKind::RawStringLit);
        assert_eq!(tokens[0].text, r#"contains "quotes""#);
    }

    #[test]
    fn test_raw_string_regex() {
        let source = r#"r"(\d+)\s+(\w+)""#;
        let tokens = lex(source);
        assert_eq!(tokens[0].kind, TokenKind::RawStringLit);
        assert_eq!(tokens[0].text, r"(\d+)\s+(\w+)");
    }

    #[test]
    fn test_raw_string_unterminated() {
        let lexer = Lexer::new(r#"r"hello"#);
        let (_, diags) = lexer.tokenize();
        assert!(diags.has_errors());
    }

    // ── Raw byte string literal tests ──────────────────────────────

    #[test]
    fn test_raw_byte_string_basic() {
        let source = r#"br"hello\nworld""#;
        let tokens = lex(source);
        assert_eq!(tokens[0].kind, TokenKind::RawByteStringLit);
        assert_eq!(tokens[0].text, r"hello\nworld");
    }

    #[test]
    fn test_raw_byte_string_with_hashes() {
        let source = r###"br#"raw "bytes""#"###;
        let tokens = lex(source);
        assert_eq!(tokens[0].kind, TokenKind::RawByteStringLit);
        assert_eq!(tokens[0].text, r#"raw "bytes""#);
    }

    // ── Identifier disambiguation tests ────────────────────────────

    #[test]
    fn test_b_as_identifier() {
        // 'b' followed by something other than '"' or '\'' should be an ident
        let tokens = lex("b + 1");
        assert_eq!(tokens[0].kind, TokenKind::Ident);
        assert_eq!(tokens[0].text, "b");
    }

    #[test]
    fn test_br_as_identifier() {
        // 'br' not followed by '"' or '#' should be an ident
        let tokens = lex("br + 1");
        assert_eq!(tokens[0].kind, TokenKind::Ident);
        assert_eq!(tokens[0].text, "br");
    }

    #[test]
    fn test_r_as_identifier() {
        // 'r' followed by something other than '"' or '#' should be an ident
        let tokens = lex("r + 1");
        assert_eq!(tokens[0].kind, TokenKind::Ident);
        assert_eq!(tokens[0].text, "r");
    }

    #[test]
    fn test_byte_string_then_ident() {
        let tokens = lex(r#"b"data" foo"#);
        assert_eq!(tokens[0].kind, TokenKind::ByteStringLit);
        assert_eq!(tokens[1].kind, TokenKind::Ident);
        assert_eq!(tokens[1].text, "foo");
    }

    #[test]
    fn test_multiple_literal_kinds() {
        let source = r#"b"bytes" r"raw" "normal" 42"#;
        let tokens = lex(source);
        assert_eq!(tokens[0].kind, TokenKind::ByteStringLit);
        assert_eq!(tokens[1].kind, TokenKind::RawStringLit);
        assert_eq!(tokens[2].kind, TokenKind::StringLit);
        assert_eq!(tokens[3].kind, TokenKind::IntLit);
    }
}
