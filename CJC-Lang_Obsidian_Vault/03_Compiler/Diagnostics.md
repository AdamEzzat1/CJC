---
title: Diagnostics
tags: [compiler, diagnostics]
status: Implemented
---

# Diagnostics

**Crate**: `cjc-diag` — `crates/cjc-diag/src/lib.rs` (~2,020 LOC).

## Summary

The error / warning / info infrastructure used by every stage of the compiler. Shared between [[Lexer]], [[Parser]], [[Type Checker]], [[HIR]], [[MIR]], and the executors.

## Key types

- `Span { start, end }` — byte offsets into source; used everywhere location matters.
- `Diagnostic { severity, code, message, primary_span, labels, notes, help }`.
- `DiagnosticBag` — an ordered collection of diagnostics (by insertion — deterministic).
- `DiagnosticBuilder` — fluent builder.
- `Severity` — `Error`, `Warning`, `Note`, `Help`.
- `ErrorCode` — a registered code from [[Error Codes]].

## Why a `Bag` instead of raising

Multiple errors can be emitted from a single pass (the parser keeps going after an error for better UX). The bag collects them all; the caller decides whether to proceed.

## Pretty printing

CLI output routes diagnostics through a formatter that underlines source spans with caret marks and colors (controllable via `--color` / `--no-color`).

## Related

- [[Error Codes]]
- [[Lexer]]
- [[Parser]]
- [[Type Checker]]
- [[CLI Surfaces]]
