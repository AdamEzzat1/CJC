//! Pratt parser for CJC source code.
//!
//! This crate implements a Pratt (top-down operator precedence) parser that
//! consumes the token stream produced by [`cjc_lexer`] and builds a complete
//! AST defined in [`cjc_ast`].
//!
//! # Quick start
//!
//! The easiest way to lex **and** parse in one step is the convenience
//! function [`parse_source`]:
//!
//! ```ignore
//! let (program, diags) = cjc_parser::parse_source("fn main() { 42 }");
//! assert!(!diags.has_errors());
//! ```
//!
//! For finer control, create a [`Parser`] from a pre-existing token vector:
//!
//! ```ignore
//! let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
//! let parser = cjc_parser::Parser::new(tokens);
//! let (program, diags) = parser.parse_program();
//! ```
//!
//! # Architecture
//!
//! The parser uses binding-power levels defined in the private [`prec`] module
//! to resolve operator precedence. The main expression driver is
//! [`Parser::parse_expr_bp`], which alternates between prefix/atom parsing
//! and an infix/postfix loop governed by the current binding power.
//!
//! Declarations (functions, structs, enums, traits, impls, imports, etc.) are
//! dispatched from [`Parser::parse_program`] via [`Parser::parse_decl`].
//!
//! Error recovery uses the [`Parser::synchronize`] method, which skips tokens
//! until a well-known synchronization point (semicolon, closing brace, or
//! declaration-starting keyword) is reached.

use cjc_ast::{
    self, BinOp, Block, CallArg, ClassDecl, ConstDecl, Decl, DeclKind, ElseBranch, EnumDecl,
    Expr, ExprKind, FieldDecl, FieldInit, FnDecl, FnSig, ForIter, ForStmt, Ident, IfStmt,
    ImplDecl, ImportDecl, LetStmt, MatchArm, Param, Pattern, PatternField, PatternKind, Program,
    RecordDecl, ShapeDim, Stmt, StmtKind, StructDecl, TraitDecl, TypeArg, TypeExpr, TypeExprKind,
    TypeParam, UnaryOp, VariantDecl, Visibility, WhileStmt,
};
use cjc_diag::{Diagnostic, DiagnosticBag};
use cjc_lexer::{Token, TokenKind};

// ── Span conversion ────────────────────────────────────────────────────

/// Convert a `cjc_diag::Span` to a `cjc_ast::Span`.
fn to_ast_span(s: cjc_diag::Span) -> cjc_ast::Span {
    cjc_ast::Span::new(s.start, s.end)
}

/// Convert a `cjc_ast::Span` to a `cjc_diag::Span`.
fn to_diag_span(s: cjc_ast::Span) -> cjc_diag::Span {
    cjc_diag::Span::new(s.start, s.end)
}

/// Merge two `cjc_ast::Span` values.
fn merge_spans(a: cjc_ast::Span, b: cjc_ast::Span) -> cjc_ast::Span {
    cjc_ast::Span::new(a.start.min(b.start), a.end.max(b.end))
}

// ── Precedence levels for Pratt parsing ────────────────────────────────

/// Binding power values used in the Pratt expression parser.
/// Each level returns `(left_bp, right_bp)`.  For left-associative
/// operators `right_bp = left_bp + 1`; for right-associative operators
/// `right_bp = left_bp`.
///
/// Higher numeric values indicate tighter binding. The gap between
/// consecutive levels leaves room for future operators without renumbering.
mod prec {
    /// Assignment `=` `+=` `-=` etc. — right-associative, lowest precedence.
    pub const ASSIGN: u8 = 2;
    /// Pipe `|>` — left-associative.
    pub const PIPE: u8 = 4;
    /// Logical or `||`.
    pub const OR: u8 = 6;
    /// Logical and `&&`.
    pub const AND: u8 = 8;
    /// Bitwise or `|`.
    pub const BIT_OR: u8 = 9;
    /// Bitwise xor `^`.
    pub const BIT_XOR: u8 = 10;
    /// Bitwise and `&`.
    pub const BIT_AND: u8 = 11;
    /// Equality `==` `!=`.
    pub const EQ: u8 = 12;
    /// Comparison `<` `>` `<=` `>=`.
    pub const CMP: u8 = 14;
    /// Shift `<<` `>>`.
    pub const SHIFT: u8 = 16;
    /// Addition `+` `-`.
    pub const ADD: u8 = 18;
    /// Multiplication `*` `/` `%`.
    pub const MUL: u8 = 20;
    /// Exponentiation `**` — right-associative.
    pub const POW: u8 = 22;
    /// Type cast `as` — left-associative, high precedence (like Rust).
    pub const AS_CAST: u8 = 23;
    /// Unary prefix `-` `!` `~`.
    pub const UNARY: u8 = 24;
    /// Postfix `.` `[` `(`.
    pub const POSTFIX: u8 = 26;
}

// ── Parser ─────────────────────────────────────────────────────────────

/// Top-down operator-precedence (Pratt) parser for CJC source code.
///
/// Construct a `Parser` with [`Parser::new`], then call [`Parser::parse_program`]
/// to consume the token stream and produce a [`Program`] AST together with any
/// accumulated [`DiagnosticBag`].
///
/// # Example
///
/// ```ignore
/// use cjc_lexer::Lexer;
/// use cjc_parser::Parser;
///
/// let (tokens, _) = Lexer::new("fn f() { 1 + 2 }").tokenize();
/// let parser = Parser::new(tokens);
/// let (program, diags) = parser.parse_program();
/// assert!(!diags.has_errors());
/// ```
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Diagnostic bag that collects parse errors and warnings.
    ///
    /// Callers can inspect this after parsing completes, but the preferred
    /// approach is to use the [`DiagnosticBag`] returned by
    /// [`Parser::parse_program`].
    pub diagnostics: DiagnosticBag,
    /// When `false`, `Ident {` is NOT parsed as a struct literal.
    /// Used to resolve the ambiguity between struct literals and block
    /// bodies in `if`/`while` conditions.
    allow_struct_lit: bool,
    /// When `false`, `|` is NOT consumed as a union-type operator in type
    /// expressions.  Used inside lambda parameter lists where `|` delimits
    /// the parameter list, not a type union.
    allow_pipe_in_type: bool,
    /// Nesting depth of while/for loops. Used to validate break/continue.
    loop_depth: usize,
}

/// Result type used internally — `Err(())` means a diagnostic was already
/// emitted and the caller should attempt recovery.
type PResult<T> = Result<T, ()>;

impl Parser {
    // ── Construction ───────────────────────────────────────────────

    /// Create a new parser from a token stream.
    ///
    /// The token vector should come from [`cjc_lexer::Lexer::tokenize`] and
    /// must end with a [`TokenKind::Eof`] sentinel (the lexer guarantees this).
    ///
    /// # Arguments
    ///
    /// * `tokens` - Complete token stream produced by the lexer.
    ///
    /// # Returns
    ///
    /// A `Parser` ready to be consumed by [`Parser::parse_program`].
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            diagnostics: DiagnosticBag::new(),
            allow_struct_lit: true,
            allow_pipe_in_type: true,
            loop_depth: 0,
        }
    }

    // ── Token stream helpers ───────────────────────────────────────

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or_else(|| self.tokens.last().unwrap())
    }

    fn peek_kind(&self) -> TokenKind {
        self.peek().kind
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.peek_kind() == kind
    }

    fn at_eof(&self) -> bool {
        self.at(TokenKind::Eof)
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: TokenKind) -> PResult<Token> {
        if self.at(kind) {
            Ok(self.advance().clone())
        } else {
            let tok = self.peek().clone();
            self.error_expected(kind.describe(), &tok);
            Err(())
        }
    }

    fn eat(&mut self, kind: TokenKind) -> Option<Token> {
        if self.at(kind) {
            Some(self.advance().clone())
        } else {
            None
        }
    }

    fn current_span(&self) -> cjc_diag::Span {
        self.peek().span
    }

    fn previous_span(&self) -> cjc_diag::Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].span
        } else {
            self.current_span()
        }
    }

    // ── Diagnostics ────────────────────────────────────────────────

    fn error(&mut self, message: impl Into<String>, span: cjc_diag::Span) {
        self.diagnostics
            .emit(Diagnostic::error("E1000", message, span));
    }

    fn error_expected(&mut self, expected: &str, found: &Token) {
        let msg = format!(
            "expected {}, found {}",
            expected,
            found.kind.describe()
        );
        self.diagnostics.emit(
            Diagnostic::error("E1001", msg, found.span)
                .with_label(found.span, format!("expected {} here", expected)),
        );
    }

    fn error_with_hint(
        &mut self,
        message: impl Into<String>,
        span: cjc_diag::Span,
        hint: impl Into<String>,
    ) {
        self.diagnostics.emit(
            Diagnostic::error("E1002", message, span).with_hint(hint),
        );
    }

    // ── Error recovery ─────────────────────────────────────────────

    /// Skip tokens until we reach a synchronization point: a semicolon,
    /// a closing brace, or a declaration-starting keyword.
    fn synchronize(&mut self) {
        loop {
            match self.peek_kind() {
                TokenKind::Eof => return,
                TokenKind::Semicolon => {
                    self.advance();
                    return;
                }
                TokenKind::RBrace => return,
                TokenKind::Struct
                | TokenKind::Class
                | TokenKind::Record
                | TokenKind::Fn
                | TokenKind::Trait
                | TokenKind::Impl
                | TokenKind::Let
                | TokenKind::Import
                | TokenKind::Mod
                | TokenKind::NoGc
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return => return,
                _ => {
                    self.advance();
                }
            }
        }
    }

    // ── Top-level entry point ──────────────────────────────────────

    /// Parse the entire token stream into a [`Program`] AST.
    ///
    /// Consume `self` and return the resulting program together with all
    /// diagnostics accumulated during parsing. The parser uses error
    /// recovery (via [`synchronize`](Self::synchronize)) so a [`Program`]
    /// is always returned, even when errors are present.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// * [`Program`] - The top-level AST containing all parsed declarations.
    /// * [`DiagnosticBag`] - Parse errors and warnings (check with
    ///   [`DiagnosticBag::has_errors`]).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (tokens, _) = cjc_lexer::Lexer::new("let x = 1;").tokenize();
    /// let (program, diags) = cjc_parser::Parser::new(tokens).parse_program();
    /// ```
    pub fn parse_program(mut self) -> (Program, DiagnosticBag) {
        let mut declarations = Vec::new();
        while !self.at_eof() {
            let before = self.pos;
            match self.parse_decl() {
                Ok(decl) => declarations.push(decl),
                Err(()) => {
                    self.synchronize();
                    // Guarantee forward progress: if neither parse_decl nor
                    // synchronize advanced the position (e.g. a stray `}`
                    // that synchronize stops at without consuming), skip
                    // the offending token so we never loop forever.
                    if self.pos == before && !self.at_eof() {
                        self.advance();
                    }
                }
            }
        }
        (Program { declarations }, self.diagnostics)
    }

    // ── Declarations ───────────────────────────────────────────────

    fn parse_decl(&mut self) -> PResult<Decl> {
        // Handle optional `pub` visibility prefix
        if self.peek_kind() == TokenKind::Pub {
            return self.parse_pub_decl();
        }
        self.parse_decl_with_vis(Visibility::Private)
    }

    /// Parse a declaration preceded by `pub`.
    fn parse_pub_decl(&mut self) -> PResult<Decl> {
        self.advance(); // consume `pub`
        self.parse_decl_with_vis(Visibility::Public)
    }

    fn parse_decl_with_vis(&mut self, vis: Visibility) -> PResult<Decl> {
        match self.peek_kind() {
            TokenKind::Struct => self.parse_struct_decl_with_vis(vis),
            TokenKind::Class => self.parse_class_decl_with_vis(vis),
            TokenKind::Record => self.parse_record_decl_with_vis(vis),
            TokenKind::Enum => self.parse_enum_decl(),
            TokenKind::At => {
                let decorators = self.parse_decorator_list()?;
                if self.peek_kind() == TokenKind::NoGc && self.peek_ahead(1) == TokenKind::Fn {
                    self.advance(); // nogc
                    self.parse_fn_decl_with_vis(true, decorators, vis)
                } else {
                    self.parse_fn_decl_with_vis(false, decorators, vis)
                }
            }
            TokenKind::Fn => self.parse_fn_decl_with_vis(false, vec![], vis),
            TokenKind::NoGc if self.peek_ahead(1) == TokenKind::Fn => self.parse_nogc_fn_decl_with_vis(vis),
            TokenKind::Trait => self.parse_trait_decl(),
            TokenKind::Impl => self.parse_impl_decl(),
            TokenKind::Import => self.parse_import_decl(),
            TokenKind::Mod => self.parse_mod_decl(),
            TokenKind::Let => self.parse_let_decl(),
            TokenKind::Const => self.parse_const_decl(),
            // Top-level statements: if, while, nogc block, expression statements
            TokenKind::If => {
                let start_span = to_ast_span(self.current_span());
                let if_stmt = self.parse_if_stmt()?;
                let end_span = if_stmt.then_block.span;
                let span = merge_spans(start_span, end_span);
                Ok(Decl {
                    kind: DeclKind::Stmt(Stmt {
                        kind: StmtKind::If(if_stmt),
                        span,
                    }),
                    span,
                })
            }
            TokenKind::While => {
                let while_stmt = self.parse_while_stmt()?;
                let span = while_stmt.body.span;
                Ok(Decl {
                    kind: DeclKind::Stmt(Stmt {
                        kind: StmtKind::While(while_stmt),
                        span,
                    }),
                    span,
                })
            }
            TokenKind::For => {
                let start_span = to_ast_span(self.current_span());
                let for_stmt = self.parse_for_stmt()?;
                let span = merge_spans(start_span, for_stmt.body.span);
                Ok(Decl {
                    kind: DeclKind::Stmt(Stmt {
                        kind: StmtKind::For(for_stmt),
                        span,
                    }),
                    span,
                })
            }
            TokenKind::NoGc if self.peek_ahead(1) == TokenKind::LBrace => {
                let nogc_start = self.advance().span;
                let block = self.parse_block()?;
                let span = merge_spans(to_ast_span(nogc_start), block.span);
                Ok(Decl {
                    kind: DeclKind::Stmt(Stmt {
                        kind: StmtKind::NoGcBlock(block),
                        span,
                    }),
                    span,
                })
            }
            _ => {
                // Try parsing as a top-level expression statement (e.g., `print("hello");`)
                let expr = self.parse_expr()?;
                let span = expr.span;
                self.expect(TokenKind::Semicolon)?;
                Ok(Decl {
                    kind: DeclKind::Stmt(Stmt {
                        kind: StmtKind::Expr(expr),
                        span,
                    }),
                    span,
                })
            }
        }
    }

    fn peek_ahead(&self, offset: usize) -> TokenKind {
        self.tokens
            .get(self.pos + offset)
            .map(|t| t.kind)
            .unwrap_or(TokenKind::Eof)
    }

    // ── struct ─────────────────────────────────────────────────────

    fn parse_struct_decl_with_vis(&mut self, vis: Visibility) -> PResult<Decl> {
        let start = self.expect(TokenKind::Struct)?.span;
        let name = self.parse_ident()?;
        let type_params = self.parse_optional_type_params()?;
        self.expect(TokenKind::LBrace)?;
        let fields = self.parse_field_list()?;
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Decl {
            kind: DeclKind::Struct(StructDecl {
                name,
                type_params,
                fields,
                vis,
            }),
            span: to_ast_span(start.merge(end)),
        })
    }

    // ── class ──────────────────────────────────────────────────────

    fn parse_class_decl_with_vis(&mut self, vis: Visibility) -> PResult<Decl> {
        let start = self.expect(TokenKind::Class)?.span;
        let name = self.parse_ident()?;
        let type_params = self.parse_optional_type_params()?;
        self.expect(TokenKind::LBrace)?;
        let fields = self.parse_field_list()?;
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Decl {
            kind: DeclKind::Class(ClassDecl {
                name,
                type_params,
                fields,
                vis,
            }),
            span: to_ast_span(start.merge(end)),
        })
    }

    // ── record ─────────────────────────────────────────────────────

    fn parse_record_decl_with_vis(&mut self, vis: Visibility) -> PResult<Decl> {
        let start = self.expect(TokenKind::Record)?.span;
        let name = self.parse_ident()?;
        let type_params = self.parse_optional_type_params()?;
        self.expect(TokenKind::LBrace)?;
        let fields = self.parse_field_list()?;
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Decl {
            kind: DeclKind::Record(RecordDecl {
                name,
                type_params,
                fields,
                vis,
            }),
            span: to_ast_span(start.merge(end)),
        })
    }

    // ── enum ───────────────────────────────────────────────────────

    fn parse_enum_decl(&mut self) -> PResult<Decl> {
        let start = self.expect(TokenKind::Enum)?.span;
        let name = self.parse_ident()?;
        let type_params = self.parse_optional_type_params()?;
        self.expect(TokenKind::LBrace)?;
        let mut variants = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let variant = self.parse_variant_decl()?;
            variants.push(variant);
            // Allow optional trailing comma
            self.eat(TokenKind::Comma);
        }
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Decl {
            kind: DeclKind::Enum(EnumDecl {
                name,
                type_params,
                variants,
            }),
            span: to_ast_span(start.merge(end)),
        })
    }

    fn parse_variant_decl(&mut self) -> PResult<VariantDecl> {
        let name = self.parse_ident()?;
        let start_span = name.span;
        let mut fields = Vec::new();
        let end_span;
        if self.eat(TokenKind::LParen).is_some() {
            // Tuple-like variant: `Variant(T1, T2, ...)`
            if !self.at(TokenKind::RParen) {
                loop {
                    fields.push(self.parse_type_expr()?);
                    if self.eat(TokenKind::Comma).is_none() {
                        break;
                    }
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                }
            }
            let rparen = self.expect(TokenKind::RParen)?;
            end_span = to_ast_span(rparen.span);
        } else {
            // Unit variant: `Variant`
            end_span = start_span;
        }
        Ok(VariantDecl {
            name,
            fields,
            span: merge_spans(start_span, end_span),
        })
    }

    // ── fields (shared between struct/class) ───────────────────────

    fn parse_field_list(&mut self) -> PResult<Vec<FieldDecl>> {
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let field = self.parse_field_decl()?;
            fields.push(field);
            // Allow optional trailing comma.
            self.eat(TokenKind::Comma);
        }
        Ok(fields)
    }

    fn parse_field_decl(&mut self) -> PResult<FieldDecl> {
        // Check for optional `pub` visibility on field
        let vis = if self.eat(TokenKind::Pub).is_some() {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type_expr()?;
        let default = if self.eat(TokenKind::Eq).is_some() {
            Some(self.parse_expr()?)
        } else {
            None
        };
        let span = merge_spans(name.span, if let Some(ref d) = default { d.span } else { ty.span });
        Ok(FieldDecl {
            name,
            ty,
            default,
            vis,
            span,
        })
    }

    // ── fn ─────────────────────────────────────────────────────────

    fn parse_fn_decl_with_vis(&mut self, is_nogc: bool, decorators: Vec<cjc_ast::Decorator>, vis: Visibility) -> PResult<Decl> {
        let start = self.expect(TokenKind::Fn)?.span;
        let name = self.parse_ident()?;
        let type_params = self.parse_optional_type_params()?;
        self.expect(TokenKind::LParen)?;
        let params = self.parse_param_list()?;
        self.expect(TokenKind::RParen)?;
        let return_type = if self.eat(TokenKind::Arrow).is_some() {
            Some(self.parse_type_expr()?)
        } else {
            None
        };
        let effect_annotation = self.parse_effect_annotation()?;
        let body = self.parse_block()?;
        let span = merge_spans(to_ast_span(start), body.span);
        Ok(Decl {
            kind: DeclKind::Fn(FnDecl {
                name,
                type_params,
                params,
                return_type,
                body,
                is_nogc,
                effect_annotation,
                decorators,
                vis,
            }),
            span,
        })
    }

    fn parse_nogc_fn_decl_with_vis(&mut self, vis: Visibility) -> PResult<Decl> {
        self.advance(); // nogc
        self.parse_fn_decl_with_vis(true, vec![], vis)
    }

    /// Parse a list of decorators: `@name` or `@name(arg1, arg2)`, one per line.
    fn parse_decorator_list(&mut self) -> PResult<Vec<cjc_ast::Decorator>> {
        let mut decorators = Vec::new();
        while self.at(TokenKind::At) {
            let at_span = self.expect(TokenKind::At)?.span;
            let name = self.parse_ident()?;
            let args = if self.eat(TokenKind::LParen).is_some() {
                let mut args = Vec::new();
                while !self.at(TokenKind::RParen) && !self.at(TokenKind::Eof) {
                    args.push(self.parse_expr()?);
                    if !self.at(TokenKind::RParen) {
                        self.expect(TokenKind::Comma)?;
                    }
                }
                self.expect(TokenKind::RParen)?;
                args
            } else {
                vec![]
            };
            let span = merge_spans(to_ast_span(at_span), name.span);
            decorators.push(cjc_ast::Decorator { name, args, span });
        }
        Ok(decorators)
    }

    /// Parse optional effect annotation: `/ pure`, `/ io`, `/ pure + alloc`, etc.
    ///
    /// Syntax: `/ effect_name (+ effect_name)*`
    /// Valid effect names: pure, io, alloc, gc, nondet, mutates, arena_ok, captures
    fn parse_effect_annotation(&mut self) -> PResult<Option<Vec<String>>> {
        if self.eat(TokenKind::Slash).is_none() {
            return Ok(None);
        }
        let mut effects = Vec::new();
        // Parse first effect name (required after `/`)
        let ident = self.parse_ident()?;
        effects.push(ident.name);
        // Parse additional effects separated by `+`
        while self.eat(TokenKind::Plus).is_some() {
            let ident = self.parse_ident()?;
            effects.push(ident.name);
        }
        Ok(Some(effects))
    }

    fn parse_param_list(&mut self) -> PResult<Vec<Param>> {
        let mut params = Vec::new();
        if self.at(TokenKind::RParen) {
            return Ok(params);
        }
        loop {
            let param = self.parse_param()?;
            let is_variadic = param.is_variadic;
            params.push(param);
            if is_variadic {
                // Variadic must be the last parameter — no more params after it.
                break;
            }
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
            // Allow trailing comma before `)`.
            if self.at(TokenKind::RParen) {
                break;
            }
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> PResult<Param> {
        // Variadic prefix: `...name: Type`
        let is_variadic = self.eat(TokenKind::DotDotDot).is_some();
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type_expr()?;
        // Optional default value: `param: Type = expr`
        let default = if self.eat(TokenKind::Eq).is_some() {
            if is_variadic {
                self.error("variadic parameters cannot have default values", self.current_span());
                return Err(());
            }
            Some(self.parse_expr()?)
        } else {
            None
        };
        let end_span = default.as_ref().map(|d| d.span).unwrap_or(ty.span);
        let span = merge_spans(name.span, end_span);
        Ok(Param { name, ty, default, is_variadic, span })
    }

    // ── trait ──────────────────────────────────────────────────────

    fn parse_trait_decl(&mut self) -> PResult<Decl> {
        let start = self.expect(TokenKind::Trait)?.span;
        let name = self.parse_ident()?;
        let type_params = self.parse_optional_type_params()?;
        let super_traits = if self.eat(TokenKind::Colon).is_some() {
            self.parse_trait_bound_list()?
        } else {
            Vec::new()
        };
        self.expect(TokenKind::LBrace)?;
        let mut methods = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let sig = self.parse_fn_sig()?;
            self.expect(TokenKind::Semicolon)?;
            methods.push(sig);
        }
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Decl {
            kind: DeclKind::Trait(TraitDecl {
                name,
                type_params,
                super_traits,
                methods,
            }),
            span: to_ast_span(start.merge(end)),
        })
    }

    fn parse_fn_sig(&mut self) -> PResult<FnSig> {
        let start = self.expect(TokenKind::Fn)?.span;
        let name = self.parse_ident()?;
        let type_params = self.parse_optional_type_params()?;
        self.expect(TokenKind::LParen)?;
        let params = self.parse_param_list()?;
        let end_paren = self.expect(TokenKind::RParen)?.span;
        let return_type = if self.eat(TokenKind::Arrow).is_some() {
            Some(self.parse_type_expr()?)
        } else {
            None
        };
        let end = return_type
            .as_ref()
            .map(|t| to_diag_span(t.span))
            .unwrap_or(end_paren);
        Ok(FnSig {
            name,
            type_params,
            params,
            return_type,
            span: to_ast_span(start.merge(end)),
        })
    }

    fn parse_trait_bound_list(&mut self) -> PResult<Vec<TypeExpr>> {
        let mut bounds = Vec::new();
        bounds.push(self.parse_type_expr()?);
        while self.eat(TokenKind::Plus).is_some() {
            bounds.push(self.parse_type_expr()?);
        }
        Ok(bounds)
    }

    // ── impl ───────────────────────────────────────────────────────

    fn parse_impl_decl(&mut self) -> PResult<Decl> {
        let start = self.expect(TokenKind::Impl)?.span;
        let type_params = self.parse_optional_type_params()?;
        let first_type = self.parse_type_expr()?;

        // P2-1: Support two syntaxes:
        //   (1) `impl Type : Trait { ... }`  — original CJC syntax
        //   (2) `impl Trait for Type { ... }` — Rust-style syntax
        let (target, trait_ref) = if self.eat(TokenKind::For).is_some() {
            // Syntax (2): first_type is the trait, next is the concrete type
            let concrete = self.parse_type_expr()?;
            (concrete, Some(first_type))
        } else if self.eat(TokenKind::Colon).is_some() {
            // Syntax (1): first_type is the target, next is the trait
            (first_type, Some(self.parse_type_expr()?))
        } else {
            // Bare impl (no trait reference)
            (first_type, None)
        };
        self.expect(TokenKind::LBrace)?;
        let mut methods = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let is_nogc = if self.at(TokenKind::NoGc) && self.peek_ahead(1) == TokenKind::Fn {
                self.advance();
                true
            } else {
                false
            };
            let fn_start = self.expect(TokenKind::Fn)?.span;
            let name = self.parse_ident()?;
            let fn_type_params = self.parse_optional_type_params()?;
            self.expect(TokenKind::LParen)?;
            let params = self.parse_param_list()?;
            self.expect(TokenKind::RParen)?;
            let return_type = if self.eat(TokenKind::Arrow).is_some() {
                Some(self.parse_type_expr()?)
            } else {
                None
            };
            let effect_annotation = self.parse_effect_annotation()?;
            let body = self.parse_block()?;
            let fn_span = merge_spans(to_ast_span(fn_start), body.span);
            methods.push(FnDecl {
                name,
                type_params: fn_type_params,
                params,
                return_type,
                body,
                is_nogc,
                effect_annotation,
                decorators: vec![],
                vis: Visibility::Private,
            });
            let _ = fn_span; // span used for method
        }
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Decl {
            kind: DeclKind::Impl(ImplDecl {
                type_params,
                target,
                trait_ref,
                methods,
                span: to_ast_span(start.merge(end)),
            }),
            span: to_ast_span(start.merge(end)),
        })
    }

    // ── import ─────────────────────────────────────────────────────

    fn parse_import_decl(&mut self) -> PResult<Decl> {
        let start = self.expect(TokenKind::Import)?.span;
        let mut path = Vec::new();
        path.push(self.parse_ident()?);
        while self.eat(TokenKind::Dot).is_some() {
            path.push(self.parse_ident()?);
        }
        let alias = if self.eat(TokenKind::As).is_some() {
            Some(self.parse_ident()?)
        } else {
            None
        };
        let end_span = alias
            .as_ref()
            .map(|a| to_diag_span(a.span))
            .or_else(|| path.last().map(|p| to_diag_span(p.span)))
            .unwrap_or(start);
        Ok(Decl {
            kind: DeclKind::Import(ImportDecl { path, alias }),
            span: to_ast_span(start.merge(end_span)),
        })
    }

    /// Parse `mod name;` — syntactic sugar for `import name`.
    fn parse_mod_decl(&mut self) -> PResult<Decl> {
        let start = self.expect(TokenKind::Mod)?.span;
        let name = self.parse_ident()?;
        let end_span = to_diag_span(name.span);
        // `mod X;` desugars to `import X`
        Ok(Decl {
            kind: DeclKind::Import(ImportDecl {
                path: vec![name],
                alias: None,
            }),
            span: to_ast_span(start.merge(end_span)),
        })
    }

    // ── let (top-level or statement) ───────────────────────────────

    fn parse_let_decl(&mut self) -> PResult<Decl> {
        let start = self.current_span();
        let let_stmt = self.parse_let_stmt()?;
        self.expect(TokenKind::Semicolon)?;
        let end = self.previous_span();
        Ok(Decl {
            kind: DeclKind::Let(let_stmt),
            span: to_ast_span(start.merge(end)),
        })
    }

    /// P2-3: Parse a compile-time constant declaration: `const NAME: Type = expr;`
    fn parse_const_decl(&mut self) -> PResult<Decl> {
        let start = self.expect(TokenKind::Const)?.span;
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type_expr()?;
        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;
        let end = self.expect(TokenKind::Semicolon)?.span;
        let span = to_ast_span(start.merge(end));
        Ok(Decl {
            kind: DeclKind::Const(ConstDecl {
                name,
                ty,
                value: Box::new(value),
                span,
            }),
            span,
        })
    }

    /// P2-5: Parse the raw content of a format string `f"..."` into segments.
    ///
    /// A segment is `(literal_text, Option<interpolated_expr>)`.
    /// The token text already has `{expr}` blocks verbatim (including the braces).
    /// We split on those blocks, re-lex and re-parse each expression fragment.
    fn parse_fstring_segments(
        &self,
        raw: &str,
        _span: cjc_ast::Span,
    ) -> PResult<Vec<(String, Option<Box<Expr>>)>> {
        let mut segments: Vec<(String, Option<Box<Expr>>)> = Vec::new();
        let bytes = raw.as_bytes();
        let mut i = 0;
        let mut literal = String::new();

        while i < bytes.len() {
            if bytes[i] == b'{' {
                // Find the matching closing `}` (respecting nesting)
                let mut depth = 1usize;
                let hole_start = i + 1;
                let mut j = hole_start;
                while j < bytes.len() && depth > 0 {
                    if bytes[j] == b'{' {
                        depth += 1;
                    } else if bytes[j] == b'}' {
                        depth -= 1;
                    }
                    j += 1;
                }
                // If we ran out of bytes without finding a closing `}`,
                // treat the `{` as a literal character (it came from `{{` escape).
                if depth > 0 {
                    literal.push('{');
                    i += 1;
                    continue;
                }
                let hole_end = j - 1; // index of the closing `}`
                let expr_src = &raw[hole_start..hole_end];

                // Re-lex and re-parse the inner expression
                let lexer = cjc_lexer::Lexer::new(expr_src);
                let (tokens, _lex_diags) = lexer.tokenize();
                let mut sub_parser = Parser::new(tokens);
                let interp_expr = sub_parser.parse_expr().map_err(|_| ())?;
                segments.push((literal.clone(), Some(Box::new(interp_expr))));
                literal.clear();
                i = j; // skip past the `}`
            } else {
                literal.push(bytes[i] as char);
                i += 1;
            }
        }

        // Trailing literal (possibly empty)
        segments.push((literal, None));
        Ok(segments)
    }

    fn parse_let_stmt(&mut self) -> PResult<LetStmt> {
        self.expect(TokenKind::Let)?;
        let mutable = self.eat(TokenKind::Mut).is_some();
        let name = self.parse_ident()?;
        let ty = if self.eat(TokenKind::Colon).is_some() {
            Some(self.parse_type_expr()?)
        } else {
            None
        };
        self.expect(TokenKind::Eq)?;
        let init = self.parse_expr()?;
        Ok(LetStmt {
            name,
            mutable,
            ty,
            init: Box::new(init),
        })
    }

    // ── Type expressions ───────────────────────────────────────────

    fn parse_type_expr(&mut self) -> PResult<TypeExpr> {
        let base = match self.peek_kind() {
            TokenKind::Ident => self.parse_named_type()?,
            TokenKind::LParen => self.parse_tuple_type()?,
            TokenKind::LBracket => self.parse_array_or_shape_type()?,
            TokenKind::Fn => self.parse_fn_type()?,
            _ => {
                let tok = self.peek().clone();
                self.error(
                    format!("expected type, found {}", tok.kind.describe()),
                    tok.span,
                );
                return Err(());
            }
        };

        // Desugar `T | null` to `Option<T>`
        // Skip this when inside lambda param lists where `|` closes the params.
        if self.allow_pipe_in_type && self.at(TokenKind::Pipe) {
            let pipe_span = self.advance().span;
            if self.at(TokenKind::Null) {
                let null_tok = self.advance();
                let base_span = base.span;
                let end_span = to_ast_span(null_tok.span);
                let option_name = cjc_ast::Ident::new("Option", base_span);
                return Ok(TypeExpr {
                    kind: TypeExprKind::Named {
                        name: option_name,
                        args: vec![TypeArg::Type(base)],
                    },
                    span: merge_spans(base_span, end_span),
                });
            } else {
                let tok = self.peek().clone();
                self.error(
                    format!(
                        "expected `null` after `|` in type expression (full union types are not yet supported), found {}",
                        tok.kind.describe()
                    ),
                    pipe_span,
                );
                return Err(());
            }
        }

        Ok(base)
    }

    fn parse_named_type(&mut self) -> PResult<TypeExpr> {
        let name = self.parse_ident()?;
        let start_span = name.span;
        let args = if self.at(TokenKind::Lt) {
            self.parse_type_arg_list()?
        } else {
            Vec::new()
        };
        let end_span = if args.is_empty() {
            start_span
        } else {
            // The `>` was consumed; use previous token span.
            to_ast_span(self.previous_span())
        };
        Ok(TypeExpr {
            kind: TypeExprKind::Named { name, args },
            span: merge_spans(start_span, end_span),
        })
    }

    fn parse_type_arg_list(&mut self) -> PResult<Vec<TypeArg>> {
        self.expect(TokenKind::Lt)?;
        let mut args = Vec::new();
        if !self.at(TokenKind::Gt) {
            loop {
                let arg = self.parse_type_arg()?;
                args.push(arg);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
                if self.at(TokenKind::Gt) {
                    break;
                }
            }
        }
        self.expect(TokenKind::Gt)?;
        Ok(args)
    }

    fn parse_type_arg(&mut self) -> PResult<TypeArg> {
        // Heuristic: if next is `[` it could be a shape, otherwise try type.
        // Integer literals become Expr type args.
        match self.peek_kind() {
            TokenKind::LBracket => {
                let dims = self.parse_shape_dims()?;
                Ok(TypeArg::Shape(dims))
            }
            TokenKind::IntLit => {
                let expr = self.parse_expr()?;
                Ok(TypeArg::Expr(expr))
            }
            _ => {
                let ty = self.parse_type_expr()?;
                Ok(TypeArg::Type(ty))
            }
        }
    }

    fn parse_shape_dims(&mut self) -> PResult<Vec<ShapeDim>> {
        self.expect(TokenKind::LBracket)?;
        let mut dims = Vec::new();
        if !self.at(TokenKind::RBracket) {
            loop {
                let dim = match self.peek_kind() {
                    TokenKind::IntLit => {
                        let tok = self.advance().clone();
                        ShapeDim::Lit(tok.int_value())
                    }
                    TokenKind::Ident => {
                        let ident = self.parse_ident()?;
                        ShapeDim::Name(ident)
                    }
                    _ => {
                        let tok = self.peek().clone();
                        self.error(
                            format!(
                                "expected shape dimension (integer or name), found {}",
                                tok.kind.describe()
                            ),
                            tok.span,
                        );
                        return Err(());
                    }
                };
                dims.push(dim);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        self.expect(TokenKind::RBracket)?;
        Ok(dims)
    }

    fn parse_tuple_type(&mut self) -> PResult<TypeExpr> {
        let start = self.expect(TokenKind::LParen)?.span;
        let mut elems = Vec::new();
        if !self.at(TokenKind::RParen) {
            loop {
                elems.push(self.parse_type_expr()?);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
                if self.at(TokenKind::RParen) {
                    break;
                }
            }
        }
        let end = self.expect(TokenKind::RParen)?.span;
        Ok(TypeExpr {
            kind: TypeExprKind::Tuple(elems),
            span: to_ast_span(start.merge(end)),
        })
    }

    fn parse_array_or_shape_type(&mut self) -> PResult<TypeExpr> {
        // `[T; N]` — array type, or `[M, N]` — shape literal.
        // Peek ahead to decide: if we see `Ident ;` or `Ident <...> ;` it
        // is an array type.  Otherwise treat as shape.
        let start = self.expect(TokenKind::LBracket)?.span;

        // Try to detect array type: first element is a type followed by `;`.
        // We save position and attempt to parse type then check for `;`.
        let saved_pos = self.pos;
        let saved_diag_len = self.diagnostics.diagnostics.len();

        if let Ok(elem_ty) = self.parse_type_expr() {
            if self.eat(TokenKind::Semicolon).is_some() {
                // This is an array type `[T; N]`.
                let size = self.parse_expr()?;
                let end = self.expect(TokenKind::RBracket)?.span;
                return Ok(TypeExpr {
                    kind: TypeExprKind::Array {
                        elem: Box::new(elem_ty),
                        size: Box::new(size),
                    },
                    span: to_ast_span(start.merge(end)),
                });
            }
        }

        // Backtrack — it is a shape literal.
        self.pos = saved_pos;
        self.diagnostics.diagnostics.truncate(saved_diag_len);

        let mut dims = Vec::new();
        if !self.at(TokenKind::RBracket) {
            loop {
                let dim = match self.peek_kind() {
                    TokenKind::IntLit => {
                        let tok = self.advance().clone();
                        ShapeDim::Lit(tok.int_value())
                    }
                    TokenKind::Ident => {
                        let ident = self.parse_ident()?;
                        ShapeDim::Name(ident)
                    }
                    _ => {
                        let tok = self.peek().clone();
                        self.error(
                            format!(
                                "expected shape dimension, found {}",
                                tok.kind.describe()
                            ),
                            tok.span,
                        );
                        return Err(());
                    }
                };
                dims.push(dim);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        let end = self.expect(TokenKind::RBracket)?.span;
        Ok(TypeExpr {
            kind: TypeExprKind::ShapeLit(dims),
            span: to_ast_span(start.merge(end)),
        })
    }

    fn parse_fn_type(&mut self) -> PResult<TypeExpr> {
        let start = self.expect(TokenKind::Fn)?.span;
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        if !self.at(TokenKind::RParen) {
            loop {
                params.push(self.parse_type_expr()?);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
                if self.at(TokenKind::RParen) {
                    break;
                }
            }
        }
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Arrow)?;
        let ret = self.parse_type_expr()?;
        let end_span = ret.span;
        Ok(TypeExpr {
            kind: TypeExprKind::Fn {
                params,
                ret: Box::new(ret),
            },
            span: merge_spans(to_ast_span(start), end_span),
        })
    }

    // ── Type parameters ────────────────────────────────────────────

    fn parse_optional_type_params(&mut self) -> PResult<Vec<TypeParam>> {
        if !self.at(TokenKind::Lt) {
            return Ok(Vec::new());
        }
        self.expect(TokenKind::Lt)?;
        let mut params = Vec::new();
        if !self.at(TokenKind::Gt) {
            loop {
                let param = self.parse_type_param()?;
                params.push(param);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
                if self.at(TokenKind::Gt) {
                    break;
                }
            }
        }
        self.expect(TokenKind::Gt)?;
        Ok(params)
    }

    fn parse_type_param(&mut self) -> PResult<TypeParam> {
        let name = self.parse_ident()?;
        let start_span = name.span;
        let bounds = if self.eat(TokenKind::Colon).is_some() {
            self.parse_trait_bound_list()?
        } else {
            Vec::new()
        };
        let end_span = bounds
            .last()
            .map(|b| b.span)
            .unwrap_or(start_span);
        Ok(TypeParam {
            name,
            bounds,
            span: merge_spans(start_span, end_span),
        })
    }

    // ── Blocks ─────────────────────────────────────────────────────

    fn parse_block(&mut self) -> PResult<Block> {
        let start = self.expect(TokenKind::LBrace)?.span;
        let mut stmts = Vec::new();
        let mut tail_expr: Option<Box<Expr>> = None;

        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            // Try to parse a statement.  If the statement is an expression
            // without a trailing semicolon *and* it is the last thing before
            // `}`, treat it as the block's tail expression.
            match self.peek_kind() {
                TokenKind::Let => {
                    let let_stmt = self.parse_let_stmt()?;
                    self.expect(TokenKind::Semicolon)?;
                    let span = merge_spans(
                        let_stmt.name.span,
                        let_stmt.init.span,
                    );
                    stmts.push(Stmt {
                        kind: StmtKind::Let(let_stmt),
                        span,
                    });
                }
                TokenKind::Return => {
                    let ret_start = self.advance().span;
                    let value = if !self.at(TokenKind::Semicolon)
                        && !self.at(TokenKind::RBrace)
                        && !self.at_eof()
                    {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    let end = self.expect(TokenKind::Semicolon)?.span;
                    stmts.push(Stmt {
                        kind: StmtKind::Return(value),
                        span: to_ast_span(ret_start.merge(end)),
                    });
                }
                TokenKind::Break => {
                    let brk_start = self.advance().span;
                    if self.loop_depth == 0 {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0400",
                            "`break` outside of loop",
                            brk_start,
                        ));
                    }
                    let end = self.expect(TokenKind::Semicolon)?.span;
                    stmts.push(Stmt {
                        kind: StmtKind::Break,
                        span: to_ast_span(brk_start.merge(end)),
                    });
                }
                TokenKind::Continue => {
                    let cont_start = self.advance().span;
                    if self.loop_depth == 0 {
                        self.diagnostics.emit(Diagnostic::error(
                            "E0401",
                            "`continue` outside of loop",
                            cont_start,
                        ));
                    }
                    let end = self.expect(TokenKind::Semicolon)?.span;
                    stmts.push(Stmt {
                        kind: StmtKind::Continue,
                        span: to_ast_span(cont_start.merge(end)),
                    });
                }
                TokenKind::If => {
                    let if_start_span = to_ast_span(self.current_span());
                    let if_stmt = self.parse_if_stmt()?;
                    let if_end_span = if_stmt.then_block.span;
                    stmts.push(Stmt {
                        kind: StmtKind::If(if_stmt),
                        span: merge_spans(if_start_span, if_end_span),
                    });
                }
                TokenKind::While => {
                    let while_stmt = self.parse_while_stmt()?;
                    let span = while_stmt.body.span;
                    stmts.push(Stmt {
                        kind: StmtKind::While(while_stmt),
                        span,
                    });
                }
                TokenKind::For => {
                    let for_start_span = to_ast_span(self.current_span());
                    let for_stmt = self.parse_for_stmt()?;
                    let span = merge_spans(for_start_span, for_stmt.body.span);
                    stmts.push(Stmt {
                        kind: StmtKind::For(for_stmt),
                        span,
                    });
                }
                TokenKind::NoGc if self.peek_ahead(1) == TokenKind::LBrace => {
                    let nogc_start = self.advance().span;
                    let block = self.parse_block()?;
                    let span = merge_spans(to_ast_span(nogc_start), block.span);
                    stmts.push(Stmt {
                        kind: StmtKind::NoGcBlock(block),
                        span,
                    });
                }
                _ => {
                    // Expression statement or tail expression.
                    let expr = self.parse_expr()?;
                    if self.eat(TokenKind::Semicolon).is_some() {
                        let span = expr.span;
                        stmts.push(Stmt {
                            kind: StmtKind::Expr(expr),
                            span,
                        });
                    } else if self.at(TokenKind::RBrace) {
                        // Tail expression.
                        tail_expr = Some(Box::new(expr));
                    } else {
                        // Missing semicolon — emit error, treat as statement.
                        let span = expr.span;
                        self.error_with_hint(
                            "expected `;` after expression statement",
                            to_diag_span(span),
                            "add a `;` here",
                        );
                        stmts.push(Stmt {
                            kind: StmtKind::Expr(expr),
                            span,
                        });
                    }
                }
            }
        }
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Block {
            stmts,
            expr: tail_expr,
            span: to_ast_span(start.merge(end)),
        })
    }

    // ── Statements ─────────────────────────────────────────────────

    fn parse_if_stmt(&mut self) -> PResult<IfStmt> {
        self.expect(TokenKind::If)?;
        // Disable struct literals in the condition so that `if x { ... }`
        // does not try to parse `x { ... }` as a struct literal.
        let prev = self.allow_struct_lit;
        self.allow_struct_lit = false;
        let condition = self.parse_expr();
        self.allow_struct_lit = prev;
        let condition = condition?;
        let then_block = self.parse_block()?;
        let else_branch = if self.eat(TokenKind::Else).is_some() {
            if self.at(TokenKind::If) {
                Some(ElseBranch::ElseIf(Box::new(self.parse_if_stmt()?)))
            } else {
                Some(ElseBranch::Else(self.parse_block()?))
            }
        } else {
            None
        };
        Ok(IfStmt {
            condition,
            then_block,
            else_branch,
        })
    }

    fn parse_while_stmt(&mut self) -> PResult<WhileStmt> {
        self.expect(TokenKind::While)?;
        // Disable struct literals in the condition so that `while x { ... }`
        // does not try to parse `x { ... }` as a struct literal.
        let prev = self.allow_struct_lit;
        self.allow_struct_lit = false;
        let condition = self.parse_expr();
        self.allow_struct_lit = prev;
        let condition = condition?;
        self.loop_depth += 1;
        let body = self.parse_block()?;
        self.loop_depth -= 1;
        Ok(WhileStmt { condition, body })
    }

    fn parse_for_stmt(&mut self) -> PResult<ForStmt> {
        self.expect(TokenKind::For)?;
        let ident = self.parse_ident()?;
        self.expect(TokenKind::In)?;

        // Parse the iterator expression. We need to detect the range form
        // `start..end`. Strategy: parse a primary expression, then check
        // if `..` follows. If so, parse the end expression. Otherwise,
        // treat the whole thing as an expression iterator.
        //
        // Disable struct literals to avoid ambiguity with the block body.
        let prev = self.allow_struct_lit;
        self.allow_struct_lit = false;
        let start_expr = self.parse_expr_bp(prec::CMP + 1)?;
        let iter = if self.eat(TokenKind::DotDot).is_some() {
            let end_expr = self.parse_expr_bp(prec::CMP + 1)?;
            ForIter::Range {
                start: Box::new(start_expr),
                end: Box::new(end_expr),
            }
        } else {
            ForIter::Expr(Box::new(start_expr))
        };
        self.allow_struct_lit = prev;

        self.loop_depth += 1;
        let body = self.parse_block()?;
        self.loop_depth -= 1;
        Ok(ForStmt { ident, iter, body })
    }

    // ── Expressions (Pratt parser) ─────────────────────────────────

    fn parse_expr(&mut self) -> PResult<Expr> {
        self.parse_expr_bp(0)
    }

    /// Pratt parser: parse an expression with minimum binding power `min_bp`.
    fn parse_expr_bp(&mut self, min_bp: u8) -> PResult<Expr> {
        // ── Prefix / atom ──────────────────────────────────────────
        let mut lhs = self.parse_prefix()?;

        // ── Infix / postfix loop ───────────────────────────────────
        loop {
            let (op_info, is_postfix) = match self.peek_kind() {
                // Postfix operators
                TokenKind::Dot => (Some((prec::POSTFIX, prec::POSTFIX + 1)), true),
                TokenKind::LParen => (Some((prec::POSTFIX, prec::POSTFIX + 1)), true),
                TokenKind::LBracket => (Some((prec::POSTFIX, prec::POSTFIX + 1)), true),
                TokenKind::Question => (Some((prec::POSTFIX, prec::POSTFIX + 1)), true),

                // Struct literal: `Ident { ... }` is only valid when lhs is
                // a bare identifier.  We need to be careful not to confuse
                // it with a block expression in statement position.  The
                // caller (parse_block) handles this ambiguity by checking
                // for a trailing semicolon.
                TokenKind::LBrace => {
                    // Only treat `{ ... }` as a struct literal when the lhs
                    // is a simple identifier AND struct literals are allowed
                    // in the current context (they are disallowed in
                    // `if`/`while` conditions to avoid ambiguity with the
                    // block body).
                    if self.allow_struct_lit && matches!(lhs.kind, ExprKind::Ident(_)) {
                        (Some((prec::POSTFIX, prec::POSTFIX + 1)), true)
                    } else {
                        break;
                    }
                }

                // Assignment (right-associative)
                TokenKind::Eq => {
                    let (l_bp, r_bp) = (prec::ASSIGN, prec::ASSIGN);
                    (Some((l_bp, r_bp)), false)
                }

                // Compound assignment (right-associative)
                TokenKind::PlusEq | TokenKind::MinusEq | TokenKind::StarEq
                | TokenKind::SlashEq | TokenKind::PercentEq | TokenKind::StarStarEq
                | TokenKind::AmpEq | TokenKind::PipeEq | TokenKind::CaretEq
                | TokenKind::LtLtEq | TokenKind::GtGtEq => {
                    (Some((prec::ASSIGN, prec::ASSIGN)), false)
                }

                // Pipe
                TokenKind::PipeGt => {
                    let (l_bp, r_bp) = (prec::PIPE, prec::PIPE + 1);
                    (Some((l_bp, r_bp)), false)
                }

                // Binary operators
                TokenKind::PipePipe => (Some((prec::OR, prec::OR + 1)), false),
                TokenKind::AmpAmp => (Some((prec::AND, prec::AND + 1)), false),
                // Bitwise operators
                TokenKind::Pipe => (Some((prec::BIT_OR, prec::BIT_OR + 1)), false),
                TokenKind::Caret => (Some((prec::BIT_XOR, prec::BIT_XOR + 1)), false),
                TokenKind::Amp => (Some((prec::BIT_AND, prec::BIT_AND + 1)), false),
                TokenKind::EqEq | TokenKind::BangEq => (Some((prec::EQ, prec::EQ + 1)), false),
                TokenKind::TildeEq | TokenKind::BangTilde => (Some((prec::EQ, prec::EQ + 1)), false),
                TokenKind::Lt | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq => {
                    (Some((prec::CMP, prec::CMP + 1)), false)
                }
                // Shift operators
                TokenKind::LtLt | TokenKind::GtGt => (Some((prec::SHIFT, prec::SHIFT + 1)), false),
                TokenKind::Plus | TokenKind::Minus => (Some((prec::ADD, prec::ADD + 1)), false),
                TokenKind::Star | TokenKind::Slash | TokenKind::Percent => {
                    (Some((prec::MUL, prec::MUL + 1)), false)
                }
                // Power (right-associative)
                TokenKind::StarStar => (Some((prec::POW, prec::POW)), false),

                // Type cast `as` — left-associative
                TokenKind::As => (Some((prec::AS_CAST, prec::AS_CAST + 1)), false),

                _ => break,
            };

            let (l_bp, r_bp) = match op_info {
                Some(bp) => bp,
                None => break,
            };

            if l_bp < min_bp {
                break;
            }

            if is_postfix {
                lhs = self.parse_postfix(lhs)?;
            } else {
                lhs = self.parse_infix(lhs, r_bp)?;
            }
        }

        Ok(lhs)
    }

    /// Parse a prefix expression or an atomic expression.
    fn parse_prefix(&mut self) -> PResult<Expr> {
        match self.peek_kind() {
            // Unary minus
            TokenKind::Minus => {
                let op_tok = self.advance().clone();
                let operand = self.parse_expr_bp(prec::UNARY)?;
                let span = merge_spans(to_ast_span(op_tok.span), operand.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            // Unary not
            TokenKind::Bang => {
                let op_tok = self.advance().clone();
                let operand = self.parse_expr_bp(prec::UNARY)?;
                let span = merge_spans(to_ast_span(op_tok.span), operand.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            // Bitwise NOT
            TokenKind::Tilde => {
                let op_tok = self.advance().clone();
                let operand = self.parse_expr_bp(prec::UNARY)?;
                let span = merge_spans(to_ast_span(op_tok.span), operand.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::BitNot,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            // If expression: `if cond { a } else { b }` used in expression context
            TokenKind::If => {
                let start_span = to_ast_span(self.current_span());
                let if_stmt = self.parse_if_stmt()?;
                let end_span = if_stmt.then_block.span;
                let span = merge_spans(start_span, end_span);
                Ok(Expr {
                    kind: ExprKind::IfExpr {
                        condition: Box::new(if_stmt.condition),
                        then_block: if_stmt.then_block,
                        else_branch: if_stmt.else_branch,
                    },
                    span,
                })
            }
            _ => self.parse_atom(),
        }
    }

    /// Parse an atomic (primary) expression.
    fn parse_atom(&mut self) -> PResult<Expr> {
        match self.peek_kind() {
            TokenKind::IntLit => {
                let tok = self.advance().clone();
                Ok(Expr {
                    kind: ExprKind::IntLit(tok.int_value()),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::FloatLit => {
                let tok = self.advance().clone();
                Ok(Expr {
                    kind: ExprKind::FloatLit(tok.float_value()),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::StringLit => {
                let tok = self.advance().clone();
                Ok(Expr {
                    kind: ExprKind::StringLit(tok.text.clone()),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::ByteStringLit => {
                let tok = self.advance().clone();
                let bytes: Vec<u8> = tok.text.chars().map(|c| c as u8).collect();
                Ok(Expr {
                    kind: ExprKind::ByteStringLit(bytes),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::ByteCharLit => {
                let tok = self.advance().clone();
                let byte_val: u8 = tok.text.parse().unwrap_or(0);
                Ok(Expr {
                    kind: ExprKind::ByteCharLit(byte_val),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::RawStringLit => {
                let tok = self.advance().clone();
                Ok(Expr {
                    kind: ExprKind::RawStringLit(tok.text.clone()),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::RawByteStringLit => {
                let tok = self.advance().clone();
                let bytes: Vec<u8> = tok.text.bytes().collect();
                Ok(Expr {
                    kind: ExprKind::RawByteStringLit(bytes),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::FStringLit => {
                let tok = self.advance().clone();
                let span = to_ast_span(tok.span);
                // Parse the raw token text into segments.
                // Format: alternating literal text and `{expr}` holes.
                let segments = self.parse_fstring_segments(&tok.text, span)?;
                Ok(Expr {
                    kind: ExprKind::FStringLit(segments),
                    span,
                })
            }
            TokenKind::RegexLit => {
                let tok = self.advance().clone();
                // Token text format: "pattern\0flags" or just "pattern" (no NUL if no flags)
                let (pattern, flags) = if let Some(idx) = tok.text.find('\0') {
                    (tok.text[..idx].to_string(), tok.text[idx + 1..].to_string())
                } else {
                    (tok.text.clone(), String::new())
                };
                Ok(Expr {
                    kind: ExprKind::RegexLit { pattern, flags },
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::True => {
                let tok = self.advance().clone();
                Ok(Expr {
                    kind: ExprKind::BoolLit(true),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::False => {
                let tok = self.advance().clone();
                Ok(Expr {
                    kind: ExprKind::BoolLit(false),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::Na => {
                let tok = self.advance().clone();
                Ok(Expr {
                    kind: ExprKind::NaLit,
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::Ident => {
                let ident = self.parse_ident()?;
                Ok(Expr {
                    kind: ExprKind::Ident(ident.clone()),
                    span: ident.span,
                })
            }
            TokenKind::Col => self.parse_col_expr(),
            TokenKind::LParen => self.parse_paren_expr(),
            TokenKind::LBracket => self.parse_array_lit(),
            TokenKind::LBracketPipe => self.parse_tensor_lit(),
            TokenKind::Pipe => self.parse_lambda(),
            TokenKind::PipePipe => self.parse_lambda_no_params(),
            TokenKind::Match => self.parse_match_expr(),
            TokenKind::LBrace => {
                let block = self.parse_block()?;
                let span = block.span;
                Ok(Expr {
                    kind: ExprKind::Block(block),
                    span,
                })
            }
            _ => {
                let tok = self.peek().clone();
                self.error(
                    format!("expected expression, found {}", tok.kind.describe()),
                    tok.span,
                );
                Err(())
            }
        }
    }

    fn parse_col_expr(&mut self) -> PResult<Expr> {
        let start = self.expect(TokenKind::Col)?.span;
        self.expect(TokenKind::LParen)?;
        let name_tok = self.expect(TokenKind::StringLit)?;
        let end = self.expect(TokenKind::RParen)?.span;
        Ok(Expr {
            kind: ExprKind::Col(name_tok.text.clone()),
            span: to_ast_span(start.merge(end)),
        })
    }

    fn parse_paren_expr(&mut self) -> PResult<Expr> {
        let start = self.expect(TokenKind::LParen)?.span;
        // Empty parens: ()
        if self.at(TokenKind::RParen) {
            let end = self.advance().span;
            return Ok(Expr {
                kind: ExprKind::TupleLit(vec![]),
                span: to_ast_span(start.merge(end)),
            });
        }
        let first = self.parse_expr()?;
        // If followed by comma, this is a tuple literal
        if self.at(TokenKind::Comma) {
            let mut elems = vec![first];
            while self.eat(TokenKind::Comma).is_some() {
                if self.at(TokenKind::RParen) {
                    break; // trailing comma
                }
                elems.push(self.parse_expr()?);
            }
            let end = self.expect(TokenKind::RParen)?.span;
            return Ok(Expr {
                kind: ExprKind::TupleLit(elems),
                span: to_ast_span(start.merge(end)),
            });
        }
        let end = self.expect(TokenKind::RParen)?.span;
        // Single expression in parens — just grouping.
        Ok(Expr {
            kind: first.kind,
            span: to_ast_span(start.merge(end)),
        })
    }

    fn parse_array_lit(&mut self) -> PResult<Expr> {
        let start = self.expect(TokenKind::LBracket)?.span;
        let mut elems = Vec::new();
        if !self.at(TokenKind::RBracket) {
            loop {
                elems.push(self.parse_expr()?);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
                if self.at(TokenKind::RBracket) {
                    break;
                }
            }
        }
        let end = self.expect(TokenKind::RBracket)?.span;
        Ok(Expr {
            kind: ExprKind::ArrayLit(elems),
            span: to_ast_span(start.merge(end)),
        })
    }

    /// Parse a tensor literal: `[| 1.0, 2.0; 3.0, 4.0 |]`
    ///
    /// Grammar: `[|` row (`;` row)* `|]`
    /// where row = expr (`,` expr)*
    ///
    /// A 1-D tensor is a single row: `[| 1.0, 2.0, 3.0 |]`
    /// A 2-D tensor uses `;` as row separator: `[| 1, 2; 3, 4 |]` (2×2)
    fn parse_tensor_lit(&mut self) -> PResult<Expr> {
        let start = self.expect(TokenKind::LBracketPipe)?.span;
        let mut rows: Vec<Vec<Expr>> = Vec::new();

        if !self.at(TokenKind::PipeRBracket) {
            loop {
                // Parse one row: comma-separated expressions
                let mut row = Vec::new();
                loop {
                    row.push(self.parse_expr()?);
                    if self.eat(TokenKind::Comma).is_none() {
                        break;
                    }
                    // Trailing comma before `;` or `|]`
                    if self.at(TokenKind::Semicolon) || self.at(TokenKind::PipeRBracket) {
                        break;
                    }
                }
                rows.push(row);
                // `;` separates rows
                if self.eat(TokenKind::Semicolon).is_none() {
                    break;
                }
                // Trailing `;` before `|]`
                if self.at(TokenKind::PipeRBracket) {
                    break;
                }
            }
        }

        let end = self.expect(TokenKind::PipeRBracket)?.span;
        Ok(Expr {
            kind: ExprKind::TensorLit { rows },
            span: to_ast_span(start.merge(end)),
        })
    }

    /// Parse a zero-parameter lambda: `|| body`.
    /// The lexer greedily tokenizes `||` as `PipePipe`, so we handle it here.
    fn parse_lambda_no_params(&mut self) -> PResult<Expr> {
        let start = to_ast_span(self.expect(TokenKind::PipePipe)?.span);
        let body = self.parse_expr()?;
        let span = merge_spans(start, body.span);
        Ok(Expr {
            kind: ExprKind::Lambda {
                params: vec![],
                body: Box::new(body),
            },
            span,
        })
    }

    /// Parse a lambda expression: `|params| body` or `|params| { block }`.
    fn parse_lambda(&mut self) -> PResult<Expr> {
        let start = to_ast_span(self.expect(TokenKind::Pipe)?.span);

        // Disable pipe-in-type so that `|x: i64|` doesn't try to parse
        // the closing `|` as a union type operator.
        let prev_pipe = self.allow_pipe_in_type;
        self.allow_pipe_in_type = false;

        // Parse parameter list (comma-separated, like fn params but delimited by `|`)
        let mut params = Vec::new();
        if !self.at(TokenKind::Pipe) {
            loop {
                let param = self.parse_param()?;
                params.push(param);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
                // Allow trailing comma before closing `|`
                if self.at(TokenKind::Pipe) {
                    break;
                }
            }
        }

        self.allow_pipe_in_type = prev_pipe;
        self.expect(TokenKind::Pipe)?;

        // Parse the body expression
        let body = self.parse_expr()?;
        let span = merge_spans(start, body.span);

        Ok(Expr {
            kind: ExprKind::Lambda {
                params,
                body: Box::new(body),
            },
            span,
        })
    }

    /// Parse a match expression: `match expr { pat => body, ... }`
    fn parse_match_expr(&mut self) -> PResult<Expr> {
        let start = self.expect(TokenKind::Match)?.span;
        // Parse the scrutinee (disable struct lit to avoid `match x { ... }`
        // being parsed as `match (x { struct lit }) { ... }`)
        let old_allow = self.allow_struct_lit;
        self.allow_struct_lit = false;
        let scrutinee = self.parse_expr()?;
        self.allow_struct_lit = old_allow;

        self.expect(TokenKind::LBrace)?;
        let mut arms = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let arm = self.parse_match_arm()?;
            arms.push(arm);
            // Arms are separated by commas (optional trailing comma)
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(scrutinee),
                arms,
            },
            span: to_ast_span(start.merge(end)),
        })
    }

    /// Parse a single match arm: `pattern => body`
    fn parse_match_arm(&mut self) -> PResult<MatchArm> {
        let pattern = self.parse_pattern()?;
        self.expect(TokenKind::FatArrow)?;
        let body = self.parse_expr()?;
        let span = merge_spans(pattern.span, body.span);
        Ok(MatchArm {
            pattern,
            body,
            span,
        })
    }

    /// Parse a pattern.
    fn parse_pattern(&mut self) -> PResult<Pattern> {
        match self.peek_kind() {
            // Wildcard `_`
            TokenKind::Underscore => {
                let tok = self.advance().clone();
                Ok(Pattern {
                    kind: PatternKind::Wildcard,
                    span: to_ast_span(tok.span),
                })
            }
            // Bool literals
            TokenKind::True => {
                let tok = self.advance().clone();
                Ok(Pattern {
                    kind: PatternKind::LitBool(true),
                    span: to_ast_span(tok.span),
                })
            }
            TokenKind::False => {
                let tok = self.advance().clone();
                Ok(Pattern {
                    kind: PatternKind::LitBool(false),
                    span: to_ast_span(tok.span),
                })
            }
            // String literal
            TokenKind::StringLit => {
                let tok = self.advance().clone();
                Ok(Pattern {
                    kind: PatternKind::LitString(tok.text.clone()),
                    span: to_ast_span(tok.span),
                })
            }
            // Integer literal (possibly negative)
            TokenKind::IntLit => {
                let tok = self.advance().clone();
                Ok(Pattern {
                    kind: PatternKind::LitInt(tok.int_value()),
                    span: to_ast_span(tok.span),
                })
            }
            // Float literal
            TokenKind::FloatLit => {
                let tok = self.advance().clone();
                Ok(Pattern {
                    kind: PatternKind::LitFloat(tok.float_value()),
                    span: to_ast_span(tok.span),
                })
            }
            // Negative literal: `-42` or `-3.14`
            TokenKind::Minus => {
                let start = self.advance().span;
                match self.peek_kind() {
                    TokenKind::IntLit => {
                        let tok = self.advance().clone();
                        Ok(Pattern {
                            kind: PatternKind::LitInt(-tok.int_value()),
                            span: to_ast_span(start.merge(tok.span)),
                        })
                    }
                    TokenKind::FloatLit => {
                        let tok = self.advance().clone();
                        Ok(Pattern {
                            kind: PatternKind::LitFloat(-tok.float_value()),
                            span: to_ast_span(start.merge(tok.span)),
                        })
                    }
                    _ => {
                        let tok = self.peek().clone();
                        self.error(
                            format!(
                                "expected numeric literal after `-` in pattern, found {}",
                                tok.kind.describe()
                            ),
                            tok.span,
                        );
                        Err(())
                    }
                }
            }
            // Tuple pattern: `(a, b, c)`
            TokenKind::LParen => {
                let start = self.advance().span;
                let mut pats = Vec::new();
                if !self.at(TokenKind::RParen) {
                    loop {
                        pats.push(self.parse_pattern()?);
                        if self.eat(TokenKind::Comma).is_none() {
                            break;
                        }
                        if self.at(TokenKind::RParen) {
                            break;
                        }
                    }
                }
                let end = self.expect(TokenKind::RParen)?.span;
                Ok(Pattern {
                    kind: PatternKind::Tuple(pats),
                    span: to_ast_span(start.merge(end)),
                })
            }
            // Identifier: could be a binding, struct destructuring, or variant pattern
            TokenKind::Ident => {
                let ident = self.parse_ident()?;
                if self.at(TokenKind::LBrace) {
                    // Struct destructuring: `Name { field, field: pat, ... }`
                    self.advance(); // `{`
                    let mut fields = Vec::new();
                    while !self.at(TokenKind::RBrace) && !self.at_eof() {
                        let field = self.parse_pattern_field()?;
                        fields.push(field);
                        if self.eat(TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                    let end = self.expect(TokenKind::RBrace)?.span;
                    Ok(Pattern {
                        kind: PatternKind::Struct {
                            name: ident.clone(),
                            fields,
                        },
                        span: merge_spans(ident.span, to_ast_span(end)),
                    })
                } else if self.at(TokenKind::LParen) {
                    // Variant pattern: `Some(x)`, `Ok(val)`, `Err(e)`
                    self.advance(); // `(`
                    let mut sub_pats = Vec::new();
                    if !self.at(TokenKind::RParen) {
                        loop {
                            sub_pats.push(self.parse_pattern()?);
                            if self.eat(TokenKind::Comma).is_none() {
                                break;
                            }
                            if self.at(TokenKind::RParen) {
                                break;
                            }
                        }
                    }
                    let end = self.expect(TokenKind::RParen)?.span;
                    Ok(Pattern {
                        kind: PatternKind::Variant {
                            enum_name: None,
                            variant: ident.clone(),
                            fields: sub_pats,
                        },
                        span: merge_spans(ident.span, to_ast_span(end)),
                    })
                } else {
                    // Plain binding pattern
                    Ok(Pattern {
                        kind: PatternKind::Binding(ident.clone()),
                        span: ident.span,
                    })
                }
            }
            _ => {
                let tok = self.peek().clone();
                self.error(
                    format!("expected pattern, found {}", tok.kind.describe()),
                    tok.span,
                );
                Err(())
            }
        }
    }

    /// Parse a single field in a struct pattern: `name` or `name: pattern`
    fn parse_pattern_field(&mut self) -> PResult<PatternField> {
        let name = self.parse_ident()?;
        let start = name.span;
        if self.eat(TokenKind::Colon).is_some() {
            let pattern = self.parse_pattern()?;
            let span = merge_spans(start, pattern.span);
            Ok(PatternField {
                name,
                pattern: Some(pattern),
                span,
            })
        } else {
            // Shorthand: `x` means `x: x`
            Ok(PatternField {
                name: name.clone(),
                pattern: None,
                span: start,
            })
        }
    }

    /// Parse a postfix operation applied to `lhs`.
    fn parse_postfix(&mut self, lhs: Expr) -> PResult<Expr> {
        match self.peek_kind() {
            TokenKind::Dot => {
                self.advance(); // `.`
                // Check for tuple field access: `expr.0`, `expr.1`, etc.
                if let TokenKind::IntLit = self.peek_kind() {
                    let tok = self.advance();
                    let idx_str = tok.text.clone();
                    let idx_span = to_ast_span(tok.span);
                    let name = Ident { name: idx_str, span: idx_span };
                    let span = merge_spans(lhs.span, idx_span);
                    return Ok(Expr {
                        kind: ExprKind::Field {
                            object: Box::new(lhs),
                            name,
                        },
                        span,
                    });
                }
                let name = self.parse_ident()?;
                let span = merge_spans(lhs.span, name.span);
                Ok(Expr {
                    kind: ExprKind::Field {
                        object: Box::new(lhs),
                        name,
                    },
                    span,
                })
            }
            TokenKind::LParen => {
                self.advance(); // `(`
                let args = self.parse_call_args()?;
                let end = self.expect(TokenKind::RParen)?.span;
                let span = merge_spans(lhs.span, to_ast_span(end));
                Ok(Expr {
                    kind: ExprKind::Call {
                        callee: Box::new(lhs),
                        args,
                    },
                    span,
                })
            }
            TokenKind::LBracket => {
                self.advance(); // `[`
                let mut indices = Vec::new();
                if !self.at(TokenKind::RBracket) {
                    loop {
                        indices.push(self.parse_expr()?);
                        if self.eat(TokenKind::Comma).is_none() {
                            break;
                        }
                        if self.at(TokenKind::RBracket) {
                            break;
                        }
                    }
                }
                let end = self.expect(TokenKind::RBracket)?.span;
                let span = merge_spans(lhs.span, to_ast_span(end));
                if indices.len() == 1 {
                    Ok(Expr {
                        kind: ExprKind::Index {
                            object: Box::new(lhs),
                            index: Box::new(indices.into_iter().next().unwrap()),
                        },
                        span,
                    })
                } else {
                    Ok(Expr {
                        kind: ExprKind::MultiIndex {
                            object: Box::new(lhs),
                            indices,
                        },
                        span,
                    })
                }
            }
            TokenKind::LBrace => {
                // Struct literal: `Name { f: v, ... }`
                // Only reached when lhs is ExprKind::Ident.
                let name = match lhs.kind {
                    ExprKind::Ident(ref id) => id.clone(),
                    _ => unreachable!("struct literal postfix should only apply to identifiers"),
                };
                self.advance(); // `{`
                let mut fields = Vec::new();
                while !self.at(TokenKind::RBrace) && !self.at_eof() {
                    let field_name = self.parse_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let value = self.parse_expr()?;
                    let field_span = merge_spans(field_name.span, value.span);
                    fields.push(FieldInit {
                        name: field_name,
                        value,
                        span: field_span,
                    });
                    if self.eat(TokenKind::Comma).is_none() {
                        break;
                    }
                }
                let end = self.expect(TokenKind::RBrace)?.span;
                let span = merge_spans(lhs.span, to_ast_span(end));
                Ok(Expr {
                    kind: ExprKind::StructLit { name, fields },
                    span,
                })
            }
            TokenKind::Question => {
                let tok = self.advance().clone();
                let span = merge_spans(lhs.span, to_ast_span(tok.span));
                Ok(Expr {
                    kind: ExprKind::Try(Box::new(lhs)),
                    span,
                })
            }
            _ => unreachable!("parse_postfix called with non-postfix token"),
        }
    }

    fn parse_call_args(&mut self) -> PResult<Vec<CallArg>> {
        let mut args = Vec::new();
        if self.at(TokenKind::RParen) {
            return Ok(args);
        }
        loop {
            let arg = self.parse_call_arg()?;
            args.push(arg);
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
            if self.at(TokenKind::RParen) {
                break;
            }
        }
        Ok(args)
    }

    fn parse_call_arg(&mut self) -> PResult<CallArg> {
        // Check for named argument: `name: expr`.
        // We look ahead: if Ident followed by Colon, it is a named arg.
        let start_span = to_ast_span(self.current_span());
        if self.at(TokenKind::Ident) && self.peek_ahead(1) == TokenKind::Colon {
            let name = self.parse_ident()?;
            self.advance(); // colon
            let value = self.parse_expr()?;
            let span = merge_spans(name.span, value.span);
            return Ok(CallArg {
                name: Some(name),
                value,
                span,
            });
        }
        let value = self.parse_expr()?;
        let span = merge_spans(start_span, value.span);
        Ok(CallArg {
            name: None,
            value,
            span,
        })
    }

    /// Parse an infix operator and its right-hand operand.
    fn parse_infix(&mut self, lhs: Expr, r_bp: u8) -> PResult<Expr> {
        let op_tok = self.advance().clone();

        match op_tok.kind {
            TokenKind::Eq => {
                let rhs = self.parse_expr_bp(r_bp)?;
                let span = merge_spans(lhs.span, rhs.span);
                Ok(Expr {
                    kind: ExprKind::Assign {
                        target: Box::new(lhs),
                        value: Box::new(rhs),
                    },
                    span,
                })
            }
            // Compound assignment: +=, -=, *=, /=, %=, **=, &=, |=, ^=, <<=, >>=
            TokenKind::PlusEq | TokenKind::MinusEq | TokenKind::StarEq
            | TokenKind::SlashEq | TokenKind::PercentEq | TokenKind::StarStarEq
            | TokenKind::AmpEq | TokenKind::PipeEq | TokenKind::CaretEq
            | TokenKind::LtLtEq | TokenKind::GtGtEq => {
                let op = compound_assign_to_binop(op_tok.kind);
                let rhs = self.parse_expr_bp(r_bp)?;
                let span = merge_spans(lhs.span, rhs.span);
                Ok(Expr {
                    kind: ExprKind::CompoundAssign {
                        op,
                        target: Box::new(lhs),
                        value: Box::new(rhs),
                    },
                    span,
                })
            }
            TokenKind::PipeGt => {
                let rhs = self.parse_expr_bp(r_bp)?;
                let span = merge_spans(lhs.span, rhs.span);
                Ok(Expr {
                    kind: ExprKind::Pipe {
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    },
                    span,
                })
            }
            TokenKind::As => {
                // Type cast: `expr as f64`, `expr as i64`, `expr as bool`, `expr as String`
                let target = self.parse_ident()?;
                let span = merge_spans(lhs.span, target.span);
                Ok(Expr {
                    kind: ExprKind::Cast {
                        expr: Box::new(lhs),
                        target_type: target,
                    },
                    span,
                })
            }
            _ => {
                let op = token_to_binop(op_tok.kind);
                let rhs = self.parse_expr_bp(r_bp)?;
                let span = merge_spans(lhs.span, rhs.span);
                Ok(Expr {
                    kind: ExprKind::Binary {
                        op,
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    },
                    span,
                })
            }
        }
    }

    // ── Identifier helper ──────────────────────────────────────────

    fn parse_ident(&mut self) -> PResult<Ident> {
        let tok = self.expect(TokenKind::Ident)?;
        Ok(Ident {
            name: tok.text.clone(),
            span: to_ast_span(tok.span),
        })
    }
}

// ── Token → BinOp mapping ──────────────────────────────────────────────

fn token_to_binop(kind: TokenKind) -> BinOp {
    match kind {
        TokenKind::Plus => BinOp::Add,
        TokenKind::Minus => BinOp::Sub,
        TokenKind::Star => BinOp::Mul,
        TokenKind::Slash => BinOp::Div,
        TokenKind::Percent => BinOp::Mod,
        TokenKind::StarStar => BinOp::Pow,
        TokenKind::EqEq => BinOp::Eq,
        TokenKind::BangEq => BinOp::Ne,
        TokenKind::Lt => BinOp::Lt,
        TokenKind::Gt => BinOp::Gt,
        TokenKind::LtEq => BinOp::Le,
        TokenKind::GtEq => BinOp::Ge,
        TokenKind::AmpAmp => BinOp::And,
        TokenKind::PipePipe => BinOp::Or,
        TokenKind::TildeEq => BinOp::Match,
        TokenKind::BangTilde => BinOp::NotMatch,
        // Bitwise
        TokenKind::Amp => BinOp::BitAnd,
        TokenKind::Pipe => BinOp::BitOr,
        TokenKind::Caret => BinOp::BitXor,
        TokenKind::LtLt => BinOp::Shl,
        TokenKind::GtGt => BinOp::Shr,
        _ => unreachable!("token_to_binop called with non-operator token {:?}", kind),
    }
}

fn compound_assign_to_binop(kind: TokenKind) -> BinOp {
    match kind {
        TokenKind::PlusEq => BinOp::Add,
        TokenKind::MinusEq => BinOp::Sub,
        TokenKind::StarEq => BinOp::Mul,
        TokenKind::SlashEq => BinOp::Div,
        TokenKind::PercentEq => BinOp::Mod,
        TokenKind::StarStarEq => BinOp::Pow,
        TokenKind::AmpEq => BinOp::BitAnd,
        TokenKind::PipeEq => BinOp::BitOr,
        TokenKind::CaretEq => BinOp::BitXor,
        TokenKind::LtLtEq => BinOp::Shl,
        TokenKind::GtGtEq => BinOp::Shr,
        _ => unreachable!("compound_assign_to_binop called with {:?}", kind),
    }
}

// ── Convenience: parse from source ─────────────────────────────────────

/// Lex and parse a CJC source string in a single call.
///
/// This is the recommended entry point for most callers. It creates a
/// [`cjc_lexer::Lexer`], tokenizes the input, feeds the tokens into a
/// [`Parser`], and returns the resulting AST with merged lexer and parser
/// diagnostics.
///
/// # Arguments
///
/// * `source` - The raw CJC source code to parse.
///
/// # Returns
///
/// A tuple of:
/// * [`Program`] - The parsed AST (may be partial when errors are present).
/// * [`DiagnosticBag`] - Combined lexer and parser diagnostics.
///
/// # Example
///
/// ```ignore
/// let (program, diags) = cjc_parser::parse_source("fn main() { 42 }");
/// assert!(!diags.has_errors());
/// assert_eq!(program.declarations.len(), 1);
/// ```
pub fn parse_source(source: &str) -> (Program, DiagnosticBag) {
    let lexer = cjc_lexer::Lexer::new(source);
    let (tokens, mut lex_diags) = lexer.tokenize();
    let parser = Parser::new(tokens);
    let (program, parse_diags) = parser.parse_program();
    // Merge diagnostics.
    for d in parse_diags.diagnostics {
        lex_diags.emit(d);
    }
    (program, lex_diags)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: lex + parse, assert no errors, return program.
    fn parse_ok(source: &str) -> Program {
        let (program, diags) = parse_source(source);
        if diags.has_errors() {
            let rendered = diags.render_all(source, "<test>");
            panic!("unexpected parse errors:\n{}", rendered);
        }
        program
    }

    /// Helper: lex + parse, assert at least one error.
    fn parse_err(source: &str) -> DiagnosticBag {
        let (_, diags) = parse_source(source);
        assert!(
            diags.has_errors(),
            "expected parse error but got none for: {}",
            source
        );
        diags
    }

    // ── Struct parsing ─────────────────────────────────────────────

    #[test]
    fn test_parse_struct_simple() {
        let prog = parse_ok("struct Point { x: f64, y: f64 }");
        assert_eq!(prog.declarations.len(), 1);
        match &prog.declarations[0].kind {
            DeclKind::Struct(s) => {
                assert_eq!(s.name.name, "Point");
                assert_eq!(s.fields.len(), 2);
                assert_eq!(s.fields[0].name.name, "x");
                assert_eq!(s.fields[1].name.name, "y");
            }
            _ => panic!("expected struct"),
        }
    }

    #[test]
    fn test_parse_struct_generic() {
        let prog = parse_ok("struct Pair<T: Clone, U> { first: T, second: U }");
        match &prog.declarations[0].kind {
            DeclKind::Struct(s) => {
                assert_eq!(s.type_params.len(), 2);
                assert_eq!(s.type_params[0].name.name, "T");
                assert_eq!(s.type_params[0].bounds.len(), 1);
                assert_eq!(s.type_params[1].name.name, "U");
                assert!(s.type_params[1].bounds.is_empty());
            }
            _ => panic!("expected struct"),
        }
    }

    // ── Class parsing ──────────────────────────────────────────────

    #[test]
    fn test_parse_class() {
        let prog = parse_ok("class Node<T> { value: T, next: Node<T> }");
        match &prog.declarations[0].kind {
            DeclKind::Class(c) => {
                assert_eq!(c.name.name, "Node");
                assert_eq!(c.type_params.len(), 1);
                assert_eq!(c.fields.len(), 2);
            }
            _ => panic!("expected class"),
        }
    }

    // ── Function parsing ───────────────────────────────────────────

    #[test]
    fn test_parse_fn_simple() {
        let prog = parse_ok("fn add(a: i64, b: i64) -> i64 { a + b }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                assert_eq!(f.name.name, "add");
                assert_eq!(f.params.len(), 2);
                assert!(f.return_type.is_some());
                assert!(!f.is_nogc);
                // The body should have a tail expression.
                assert!(f.body.expr.is_some());
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_fn_nogc() {
        let prog = parse_ok("nogc fn fast(x: f64) -> f64 { x }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                assert!(f.is_nogc);
                assert_eq!(f.name.name, "fast");
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_fn_no_return_type() {
        let prog = parse_ok("fn greet(name: String) { name }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                assert!(f.return_type.is_none());
            }
            _ => panic!("expected fn"),
        }
    }

    // ── Trait parsing ──────────────────────────────────────────────

    #[test]
    fn test_parse_trait() {
        let prog = parse_ok(
            "trait Numeric: Add + Mul { fn zero() -> Self; fn one() -> Self; }",
        );
        match &prog.declarations[0].kind {
            DeclKind::Trait(t) => {
                assert_eq!(t.name.name, "Numeric");
                assert_eq!(t.super_traits.len(), 2);
                assert_eq!(t.methods.len(), 2);
                assert_eq!(t.methods[0].name.name, "zero");
            }
            _ => panic!("expected trait"),
        }
    }

    // ── Impl parsing ──────────────────────────────────────────────

    #[test]
    fn test_parse_impl() {
        let prog = parse_ok(
            "impl<T> Vec<T> : Iterable { fn len(self: Vec<T>) -> i64 { 0 } }",
        );
        match &prog.declarations[0].kind {
            DeclKind::Impl(i) => {
                assert_eq!(i.type_params.len(), 1);
                assert!(i.trait_ref.is_some());
                assert_eq!(i.methods.len(), 1);
            }
            _ => panic!("expected impl"),
        }
    }

    // ── Import parsing ─────────────────────────────────────────────

    #[test]
    fn test_parse_import() {
        let prog = parse_ok("import std.io.File as F");
        match &prog.declarations[0].kind {
            DeclKind::Import(i) => {
                assert_eq!(i.path.len(), 3);
                assert_eq!(i.path[0].name, "std");
                assert_eq!(i.path[1].name, "io");
                assert_eq!(i.path[2].name, "File");
                assert_eq!(i.alias.as_ref().unwrap().name, "F");
            }
            _ => panic!("expected import"),
        }
    }

    #[test]
    fn test_parse_import_no_alias() {
        let prog = parse_ok("import math.linalg");
        match &prog.declarations[0].kind {
            DeclKind::Import(i) => {
                assert_eq!(i.path.len(), 2);
                assert!(i.alias.is_none());
            }
            _ => panic!("expected import"),
        }
    }

    // ── Let statement ──────────────────────────────────────────────

    #[test]
    fn test_parse_let() {
        let prog = parse_ok("let x: i64 = 42;");
        match &prog.declarations[0].kind {
            DeclKind::Let(l) => {
                assert_eq!(l.name.name, "x");
                assert!(!l.mutable);
                assert!(l.ty.is_some());
            }
            _ => panic!("expected let"),
        }
    }

    #[test]
    fn test_parse_let_mut() {
        let prog = parse_ok("let mut count = 0;");
        match &prog.declarations[0].kind {
            DeclKind::Let(l) => {
                assert!(l.mutable);
                assert!(l.ty.is_none());
            }
            _ => panic!("expected let"),
        }
    }

    // ── Expression parsing ─────────────────────────────────────────

    #[test]
    fn test_parse_binary_precedence() {
        // `1 + 2 * 3` should parse as `1 + (2 * 3)`.
        let prog = parse_ok("fn main() { 1 + 2 * 3 }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                match &tail.kind {
                    ExprKind::Binary { op, left, right } => {
                        assert_eq!(*op, BinOp::Add);
                        // left should be 1.
                        assert!(matches!(left.kind, ExprKind::IntLit(1)));
                        // right should be 2 * 3.
                        match &right.kind {
                            ExprKind::Binary { op, .. } => assert_eq!(*op, BinOp::Mul),
                            _ => panic!("expected binary mul"),
                        }
                    }
                    _ => panic!("expected binary add"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_unary() {
        let prog = parse_ok("fn f() { -x }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                match &tail.kind {
                    ExprKind::Unary { op, .. } => assert_eq!(*op, UnaryOp::Neg),
                    _ => panic!("expected unary"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_call_with_named_args() {
        let prog = parse_ok("fn f() { create(width: 10, height: 20) }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                match &tail.kind {
                    ExprKind::Call { args, .. } => {
                        assert_eq!(args.len(), 2);
                        assert_eq!(args[0].name.as_ref().unwrap().name, "width");
                        assert_eq!(args[1].name.as_ref().unwrap().name, "height");
                    }
                    _ => panic!("expected call"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_field_access_and_method_call() {
        let prog = parse_ok("fn f() { obj.field.method(x) }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                // Should be: Call { callee: Field { object: Field { ... }, name: method }, args: [x] }
                match &tail.kind {
                    ExprKind::Call { callee, args } => {
                        assert_eq!(args.len(), 1);
                        match &callee.kind {
                            ExprKind::Field { name, .. } => {
                                assert_eq!(name.name, "method");
                            }
                            _ => panic!("expected field access"),
                        }
                    }
                    _ => panic!("expected call"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_index_and_multi_index() {
        let prog = parse_ok("fn f() { a[0]; b[1, 2] }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                // First statement: a[0] — single index.
                match &f.body.stmts[0].kind {
                    StmtKind::Expr(e) => match &e.kind {
                        ExprKind::Index { .. } => {}
                        _ => panic!("expected index"),
                    },
                    _ => panic!("expected expr stmt"),
                }
                // Tail expression: b[1, 2] — multi-index.
                let tail = f.body.expr.as_ref().unwrap();
                match &tail.kind {
                    ExprKind::MultiIndex { indices, .. } => {
                        assert_eq!(indices.len(), 2);
                    }
                    _ => panic!("expected multi-index"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_pipe() {
        let prog = parse_ok("fn f() { data |> filter(x) |> map(y) }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                // Should be left-associative: (data |> filter(x)) |> map(y)
                match &tail.kind {
                    ExprKind::Pipe { right, .. } => {
                        match &right.kind {
                            ExprKind::Call { callee, .. } => match &callee.kind {
                                ExprKind::Ident(id) => assert_eq!(id.name, "map"),
                                _ => panic!("expected ident"),
                            },
                            _ => panic!("expected call"),
                        }
                    }
                    _ => panic!("expected pipe"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_assignment() {
        let prog = parse_ok("fn f() { x = 10; }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => match &f.body.stmts[0].kind {
                StmtKind::Expr(e) => match &e.kind {
                    ExprKind::Assign { .. } => {}
                    _ => panic!("expected assign"),
                },
                _ => panic!("expected expr stmt"),
            },
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_struct_literal() {
        let prog = parse_ok("fn f() { Point { x: 1, y: 2 } }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                match &tail.kind {
                    ExprKind::StructLit { name, fields } => {
                        assert_eq!(name.name, "Point");
                        assert_eq!(fields.len(), 2);
                    }
                    _ => panic!("expected struct lit"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_array_literal() {
        let prog = parse_ok("fn f() { [1, 2, 3] }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                match &tail.kind {
                    ExprKind::ArrayLit(elems) => assert_eq!(elems.len(), 3),
                    _ => panic!("expected array lit"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_col() {
        let prog = parse_ok(r#"fn f() { col("price") }"#);
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                match &tail.kind {
                    ExprKind::Col(name) => assert_eq!(name, "price"),
                    _ => panic!("expected col"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    // ── Control flow ───────────────────────────────────────────────

    #[test]
    fn test_parse_if_else_if_else() {
        let prog = parse_ok(
            "fn f() { if x { 1; } else if y { 2; } else { 3; } }",
        );
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                assert_eq!(f.body.stmts.len(), 1);
                match &f.body.stmts[0].kind {
                    StmtKind::If(if_stmt) => {
                        assert!(if_stmt.else_branch.is_some());
                        match if_stmt.else_branch.as_ref().unwrap() {
                            ElseBranch::ElseIf(elif) => {
                                assert!(elif.else_branch.is_some());
                                match elif.else_branch.as_ref().unwrap() {
                                    ElseBranch::Else(_) => {}
                                    _ => panic!("expected else block"),
                                }
                            }
                            _ => panic!("expected else-if"),
                        }
                    }
                    _ => panic!("expected if"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_while() {
        let prog = parse_ok("fn f() { while x > 0 { x = x - 1; } }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => match &f.body.stmts[0].kind {
                StmtKind::While(w) => {
                    assert!(!w.body.stmts.is_empty());
                }
                _ => panic!("expected while"),
            },
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_return() {
        let prog = parse_ok("fn f() { return 42; }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => match &f.body.stmts[0].kind {
                StmtKind::Return(Some(e)) => {
                    assert!(matches!(e.kind, ExprKind::IntLit(42)));
                }
                _ => panic!("expected return"),
            },
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_nogc_block() {
        let prog = parse_ok("fn f() { nogc { x + y; } }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => match &f.body.stmts[0].kind {
                StmtKind::NoGcBlock(block) => {
                    assert_eq!(block.stmts.len(), 1);
                }
                _ => panic!("expected nogc block"),
            },
            _ => panic!("expected fn"),
        }
    }

    // ── Error recovery ─────────────────────────────────────────────

    #[test]
    fn test_error_recovery_missing_semicolon() {
        // Missing semicolon after let — parser should recover and parse
        // the next declaration.
        let (prog, diags) = parse_source("let x = 1\nfn f() { 0 }");
        assert!(diags.has_errors());
        // Should still have parsed the fn.
        assert!(prog.declarations.iter().any(|d| matches!(&d.kind, DeclKind::Fn(_))));
    }

    #[test]
    fn test_error_recovery_unexpected_token() {
        let diags = parse_err("@@@ fn f() { 0 }");
        assert!(diags.has_errors());
    }

    #[test]
    fn test_error_expected_expression() {
        let diags = parse_err("fn f() { let x = ; }");
        assert!(diags.has_errors());
    }

    // ── Complex integration test ───────────────────────────────────

    #[test]
    fn test_parse_full_program() {
        let source = r#"
            import std.math as m

            struct Vec2 {
                x: f64,
                y: f64
            }

            fn dot(a: Vec2, b: Vec2) -> f64 {
                a.x * b.x + a.y * b.y
            }

            trait Shape {
                fn area(self: Self) -> f64;
            }

            impl Vec2 : Shape {
                fn area(self: Vec2) -> f64 {
                    self.x * self.y
                }
            }

            let result: f64 = dot(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 });
        "#;
        let prog = parse_ok(source);
        assert_eq!(prog.declarations.len(), 6);
    }

    #[test]
    fn test_parse_pipe_chain() {
        let source = r#"
            fn pipeline(data: DataFrame) -> DataFrame {
                data
                    |> filter(col("age") > 18)
                    |> group_by(col("city"))
            }
        "#;
        // Should parse without errors. The pipe chain produces a tail
        // expression.
        let prog = parse_ok(source);
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                assert!(f.body.expr.is_some());
            }
            _ => panic!("expected fn"),
        }
    }

    // ── Boolean and logical operators ──────────────────────────────

    #[test]
    fn test_parse_logical_operators() {
        let prog = parse_ok("fn f() { a && b || c }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                // `&&` binds tighter than `||`, so: (a && b) || c
                match &tail.kind {
                    ExprKind::Binary { op, .. } => assert_eq!(*op, BinOp::Or),
                    _ => panic!("expected binary or"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn test_parse_comparison_chain() {
        let prog = parse_ok("fn f() { x == 1 && y != 2 }");
        match &prog.declarations[0].kind {
            DeclKind::Fn(f) => {
                let tail = f.body.expr.as_ref().unwrap();
                match &tail.kind {
                    ExprKind::Binary { op, .. } => assert_eq!(*op, BinOp::And),
                    _ => panic!("expected and"),
                }
            }
            _ => panic!("expected fn"),
        }
    }
}
