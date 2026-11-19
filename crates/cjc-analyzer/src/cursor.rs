//! Postulate Phase 2 — what is under the cursor, answered by the compiler.
//!
//! `postulate/cursorContext` tells a companion HUD which panels matter at a
//! cursor position. Every field comes from CJC's own lexer and parser, or from
//! the cached run's `GraphExport` spans; nothing is guessed from text:
//!
//! - `symbol`: the identifier token under the cursor (`cjc-lexer`).
//! - `function`: the innermost `fn` whose body contains the cursor (`cjc-parser`).
//! - `training_loop`: the innermost `while`/`for` containing the cursor whose
//!   body calls `.backward(` (lexer tokens inside the loop's AST span).
//! - `nodes`: GradGraph nodes whose recorded span contains the cursor, or
//!   `null` when this document version has not been run yet. A cursor move
//!   never starts a run.

use cjc_ast::{Block, DeclKind, ElseBranch, IfStmt, Program, Stmt, StmtKind};
use cjc_lexer::{Lexer, Token, TokenKind};
use serde_json::{json, Value as Json};

use crate::postulate::{position_of, RunCapture};

/// A byte range `[start, end)` in the document.
type Range = (usize, usize);

fn contains(r: Range, offset: usize) -> bool {
    r.0 <= offset && offset <= r.1
}

/// LSP position (0-based line, UTF-16 column) → byte offset, clamped to the text.
pub fn offset_of(src: &str, line: u32, character: u32) -> usize {
    let mut start = 0;
    for _ in 0..line {
        match src[start..].find('\n') {
            Some(i) => start += i + 1,
            None => return src.len(),
        }
    }
    let end = src[start..].find('\n').map_or(src.len(), |i| start + i);
    let mut units = 0u32;
    for (i, c) in src[start..end].char_indices() {
        if units >= character {
            return start + i;
        }
        units += c.len_utf16() as u32;
    }
    end
}

fn collect_loops_in_block(b: &Block, out: &mut Vec<(Range, Range)>) {
    for s in &b.stmts {
        collect_loops(s, out);
    }
}

fn collect_loops_in_if(i: &IfStmt, out: &mut Vec<(Range, Range)>) {
    collect_loops_in_block(&i.then_block, out);
    match &i.else_branch {
        Some(ElseBranch::ElseIf(e)) => collect_loops_in_if(e, out),
        Some(ElseBranch::Else(b)) => collect_loops_in_block(b, out),
        None => {}
    }
}

/// Every loop statement: (whole statement span, body span).
fn collect_loops(s: &Stmt, out: &mut Vec<(Range, Range)>) {
    match &s.kind {
        StmtKind::While(w) => {
            out.push(((s.span.start, s.span.end), (w.body.span.start, w.body.span.end)));
            collect_loops_in_block(&w.body, out);
        }
        StmtKind::For(f) => {
            out.push(((s.span.start, s.span.end), (f.body.span.start, f.body.span.end)));
            collect_loops_in_block(&f.body, out);
        }
        StmtKind::If(i) => collect_loops_in_if(i, out),
        StmtKind::NoGcBlock(b) => collect_loops_in_block(b, out),
        _ => {}
    }
}

/// (name, body span) of every function, including impl methods.
fn functions(p: &Program) -> Vec<(String, Range)> {
    let mut out = Vec::new();
    for d in &p.declarations {
        match &d.kind {
            DeclKind::Fn(f) => out.push((f.name.name.clone(), (f.body.span.start, f.body.span.end))),
            DeclKind::Impl(i) => {
                for f in &i.methods {
                    out.push((f.name.name.clone(), (f.body.span.start, f.body.span.end)));
                }
            }
            _ => {}
        }
    }
    out
}

fn loops(p: &Program) -> Vec<(Range, Range)> {
    let mut out = Vec::new();
    for d in &p.declarations {
        match &d.kind {
            DeclKind::Fn(f) => collect_loops_in_block(&f.body, &mut out),
            DeclKind::Impl(i) => i.methods.iter().for_each(|f| collect_loops_in_block(&f.body, &mut out)),
            DeclKind::Stmt(s) => collect_loops(s, &mut out),
            _ => {}
        }
    }
    out
}

/// Does `.backward(` occur as tokens inside `range`?
fn calls_backward(tokens: &[Token], range: Range) -> bool {
    tokens.windows(3).any(|w| {
        w[0].kind == TokenKind::Dot
            && w[1].kind == TokenKind::Ident
            && w[1].text == "backward"
            && w[2].kind == TokenKind::LParen
            && range.0 <= w[0].span.start
            && w[2].span.end <= range.1
    })
}

fn lsp_range(src: &str, r: Range) -> Json {
    let (s, e) = (position_of(src, r.0), position_of(src, r.1));
    json!({ "start": {"line": s.line, "character": s.character},
            "end": {"line": e.line, "character": e.character} })
}

/// The identifier token whose span contains `offset`, if any.
pub fn ident_at(tokens: &[Token], offset: usize) -> Option<&Token> {
    tokens
        .iter()
        .find(|t| t.kind == TokenKind::Ident && t.span.start <= offset && offset <= t.span.end)
}

pub fn tokens(src: &str) -> Vec<Token> {
    Lexer::new(src).tokenize().0
}

/// The `postulate/cursorContext` body. `run` is the cached run of this exact
/// document version, if there is one.
pub fn cursor_context(src: &str, line: u32, character: u32, run: Option<&RunCapture>) -> Json {
    let offset = offset_of(src, line, character);
    let toks = tokens(src);
    let (program, _) = cjc_parser::parse_source(src);

    let symbol = ident_at(&toks, offset).map(|t| t.text.clone());
    let function = functions(&program)
        .into_iter()
        .filter(|(_, r)| contains(*r, offset))
        .min_by_key(|(_, r)| r.1 - r.0)
        .map(|(n, _)| n);
    let training_loop = loops(&program)
        .into_iter()
        .filter(|(whole, body)| contains(*whole, offset) && calls_backward(&toks, *body))
        .min_by_key(|(whole, _)| whole.1 - whole.0)
        .map(|(whole, _)| lsp_range(src, whole));
    let nodes = run.map(|r| {
        r.eval
            .graph
            .as_ref()
            .map(|g| {
                g.nodes
                    .iter()
                    .filter(|n| n.span.is_some_and(|s| (s.start as usize) <= offset && offset <= s.end as usize))
                    .map(|n| json!({ "id": n.id, "op": n.op, "shape": n.shape }))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });

    json!({
        "line": line,
        "character": character,
        "offset": offset,
        "symbol": symbol,
        "function": function,
        "training_loop": training_loop,
        "nodes": nodes,
    })
}

/// Occurrences of the identifier under the cursor (`textDocument/documentHighlight`).
pub fn highlights(src: &str, line: u32, character: u32) -> Vec<lsp_types::DocumentHighlight> {
    let offset = offset_of(src, line, character);
    let toks = tokens(src);
    let Some(name) = ident_at(&toks, offset).map(|t| t.text.clone()) else { return Vec::new() };
    toks.iter()
        .filter(|t| t.kind == TokenKind::Ident && t.text == name)
        .map(|t| lsp_types::DocumentHighlight {
            range: lsp_types::Range { start: position_of(src, t.span.start), end: position_of(src, t.span.end) },
            kind: Some(lsp_types::DocumentHighlightKind::TEXT),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "fn helper(x: f64) -> f64 {\n    x * 2.0\n}\nlet i = 0;\nwhile i < 3 {\n    let g = GradGraph.new();\n    let w = g.parameter(Tensor.zeros([2]));\n    g.backward(g.sum(w));\n    i = i + 1;\n}\nwhile i < 6 {\n    i = i + 1;\n}\n";

    #[test]
    fn offsets_round_trip_through_positions() {
        for (b, _) in SRC.char_indices() {
            let p = position_of(SRC, b);
            assert_eq!(offset_of(SRC, p.line, p.character), b);
        }
    }

    #[test]
    fn training_loop_is_the_loop_that_calls_backward() {
        let c = cursor_context(SRC, 6, 12, None); // the `g` of `g.parameter`
        assert_eq!(c["symbol"], "g");
        assert_eq!(c["training_loop"]["start"]["line"], 4);
        assert_eq!(c["training_loop"]["end"]["line"], 9);
        assert_eq!(c["nodes"], Json::Null, "no run: nodes unknown, not empty");
        // The second loop has no backward call.
        assert_eq!(cursor_context(SRC, 11, 4, None)["training_loop"], Json::Null);
        assert_eq!(cursor_context(SRC, 1, 4, None)["function"], "helper");
        assert_eq!(cursor_context(SRC, 6, 4, None)["function"], Json::Null);
    }

    #[test]
    fn highlights_every_occurrence_of_the_identifier() {
        let h = highlights(SRC, 7, 4); // `g` in `g.backward`
        assert_eq!(h.len(), 4, "{h:?}"); // let g, g.parameter, g.backward, g.sum
    }
}
