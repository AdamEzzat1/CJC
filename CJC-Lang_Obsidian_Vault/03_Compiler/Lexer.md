---
title: Lexer
tags: [compiler, frontend]
status: Implemented
---

# Lexer

**Crate**: `cjc-lexer` — `crates/cjc-lexer/src/lib.rs` (~1,949 LOC).

## Summary

The lexer tokenizes CJC-Lang source into a stream of `Token { kind: TokenKind, span, text }` values. It handles all literal forms, keywords, operators, and string-family literals (plain, format, raw, byte, regex).

## Public API

```rust
let (tokens, diags) = Lexer::new(src).tokenize();
```

Returns a tuple of `(Vec<Token>, DiagnosticBag)`. Lex errors go into the [[Diagnostics]] bag rather than panicking.

## TokenKind variants

~30+ variants including:

- Literals: `IntLit`, `FloatLit`, `StringLit`, `ByteStringLit`, `RawStringLit`, `FormatStringLit`, `RegexLit`
- Identifier: `Ident`
- Keywords: `Fn`, `Let`, `If`, `Else`, `While`, `For`, `In`, `Return`, `Struct`, `Enum`, `Class`, `Trait`, `Match`, `Mod`, `Import`, `Pub`, ...
- Operators: `Plus`, `Minus`, `Star`, `Slash`, `StarStar`, `Eq`, `PlusEq`, ...
- Punctuation: `LParen`, `RParen`, `LBrace`, `RBrace`, `LBracket`, `RBracket`, `Comma`, `Semi`, `Colon`, ...
- Special: `PipeArrow` (`|>`), `FatArrow` (`=>`), `Arrow` (`->`)

## Tensor literal lexing

The token `[|` and `|]` are recognized as dedicated tokens for tensor literals, distinct from a `[` followed by a `|`.

## Determinism

The lexer is pure and position-preserving. Spans carry byte offsets into the source, which flow through [[Diagnostics]] so error messages point at exact source locations.

## Source

- `crates/cjc-lexer/src/lib.rs` — all logic in one file.

## Related

- [[Parser]]
- [[AST]]
- [[Diagnostics]]
- [[Syntax]]
- [[Format Strings]]
